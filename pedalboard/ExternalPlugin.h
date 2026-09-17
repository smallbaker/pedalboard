/*
 * pedalboard
 * Copyright 2021 Spotify AB
 *
 * Licensed under the GNU Public License, Version 3.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    https://www.gnu.org/licenses/gpl-3.0.html
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <cctype>
#include <map>
#include <mutex>
#include <optional>
#include <tuple>
#include <utility>

#include "JuceHeader.h"
#if JUCE_LINUX
#include <sys/utsname.h>
#endif

#include "AudioUnitParser.h"
#include "Plugin.h"
#include <pybind11/stl.h>

#include "juce_overrides/juce_PatchedVST3PluginFormat.h"
#include "process.h"

#if JUCE_MAC
#include <AudioToolbox/AudioUnitUtilities.h>
#endif

namespace Pedalboard {

// JUCE external plugins use some global state; here we lock that state
// to play nicely with the Python interpreter.
static std::mutex EXTERNAL_PLUGIN_MUTEX;
static int NUM_ACTIVE_EXTERNAL_PLUGINS = 0;

static const float DEFAULT_INITIALIZATION_TIMEOUT_SECONDS = 10.0f;

static const std::string AUDIO_UNIT_NOT_INSTALLED_ERROR =
    "macOS requires plugin files to be moved to "
    "/Library/Audio/Plug-Ins/Components/ or "
    "~/Library/Audio/Plug-Ins/Components/ before loading.";

static constexpr const char *EXTERNAL_PLUGIN_PROCESS_DOCSTRING = R"(
Pass a buffer of audio (as a 32- or 64-bit NumPy array) *or* a list of
MIDI messages to this plugin, returning audio.

(If calling this multiple times with multiple effect plugins, consider
creating a :class:`pedalboard.Pedalboard` object instead.)

When provided audio as input, the returned array may contain up to (but not
more than) the same number of samples as were provided. If fewer samples
were returned than expected, the plugin has likely buffered audio inside
itself. To receive the remaining audio, pass another audio buffer into 
``process`` with ``reset`` set to ``True``.

If the provided buffer uses a 64-bit datatype, it will be converted to 32-bit
for processing.

If provided MIDI messages as input, the provided ``midi_messages`` must be
a Python ``List`` containing one of the following types:

 - Objects with a ``bytes()`` method and ``time`` property (such as :doc:`mido:messages`
   from :doc:`mido:index`, not included with Pedalboard)
 - Tuples that look like: ``(midi_bytes: bytes, timestamp_in_seconds: float)``
 - Tuples that look like: ``(midi_bytes: List[int], timestamp_in_seconds: float)``

The returned array will contain ``duration`` seconds worth of audio at the
provided ``sample_rate``.

Each MIDI message will be sent to the plugin at its
timestamp, where a timestamp of ``0`` indicates the start of the buffer, and
a timestamp equal to ``duration`` indicates the end of the buffer. (Any MIDI
messages whose timestamps are greater than ``duration`` will be ignored.)

The provided ``buffer_size`` argument will be used to control the size of
each chunk of audio returned by the plugin at once. Higher buffer sizes may
speed up processing, but may cause increased memory usage.

The ``reset`` flag determines if this plugin should be reset before
processing begins, clearing any state from previous calls to ``process``.
If calling ``process`` multiple times while processing the same audio or
MIDI stream, set ``reset`` to ``False``.

.. note::
    The :py:meth:`process` method can also be used via :py:meth:`__call__`;
    i.e.: just calling this object like a function (``my_plugin(...)``) will
    automatically invoke :py:meth:`process` with the same arguments.


Examples
--------

Running audio through an external effect plugin::

   from pedalboard import load_plugin
   from pedalboard.io import AudioFile

   plugin = load_plugin("../path-to-my-plugin-file")
   assert plugin.is_effect
   with AudioFile("input-audio.wav") as f:
       output_audio = plugin(f.read(), f.samplerate)


Rendering MIDI via an external instrument plugin::

   from pedalboard import load_plugin
   from pedalboard.io import AudioFile
   from mido import Message # not part of Pedalboard, but convenient!

   plugin = load_plugin("../path-to-my-plugin-file")
   assert plugin.is_instrument

   sample_rate = 44100
   num_channels = 2
   with AudioFile("output-audio.wav", "w", sample_rate, num_channels) as f:
       f.write(plugin(
           [Message("note_on", note=60), Message("note_off", note=60, time=4)],
           sample_rate=sample_rate,
           duration=5,
           num_channels=num_channels
       ))


*Support for instrument plugins introduced in v0.7.4.*
          )";

static constexpr const char *SHOW_EDITOR_DOCSTRING = R"(
Show the UI of this plugin in a window with its own title bar.

``title`` names the window (default: the plugin's name). ``powered`` sets the
state of the power button in the title bar, and ``on_power`` is called with
the new state (True = on, False = bypassed) whenever the button is clicked;
it runs on the thread that called this method, from inside the event loop.

This method may only be called on the main thread, and will block
the main thread until any of the following things happens:

 - the window is closed by clicking the close button
 - the window is closed by pressing the appropriate (OS-specific) keyboard shortcut
 - a KeyboardInterrupt (Ctrl-C) is sent to the program
 - the :py:meth:`threading.Event.set` method is called (by another thread)
   on a provided :py:class:`threading.Event` object

An example of how to programmatically close an editor window::

   import pedalboard
   from threading import Event, Thread

   plugin = pedalboard.load_plugin("../path-to-my-plugin-file")
   close_window_event = Event()

   def other_thread():
       # do something to determine when to close the window
       if should_close_window:
           close_window_event.set()

   thread = Thread(target=other_thread)
   thread.start()

   # This will block until the other thread calls .set():
   plugin.show_editor(close_window_event)
)";

inline std::vector<std::string> findInstalledVSTPluginPaths() {
  // Ensure we have a MessageManager, which is required by the VST wrapper
  // Without this, we get an assert(false) from JUCE at runtime
  juce::MessageManager::getInstance();
  juce::PatchedVST3PluginFormat format;
  std::vector<std::string> pluginPaths;
  for (juce::String pluginIdentifier : format.searchPathsForPlugins(
           format.getDefaultLocationsToSearch(), true, false)) {
    pluginPaths.push_back(
        format.getNameOfPluginFromIdentifier(pluginIdentifier).toStdString());
  }
  return pluginPaths;
}

/**
 * Given a py::object representing a Python list object filled with Tuple[bytes,
 * float], each representing a MIDI message at a specific timestamp, return a
 * juce::MidiBuffer.
 */
juce::MidiBuffer parseMidiBufferFromPython(py::object midiMessages,
                                           float sampleRate) {
  juce::MidiBuffer buf;
  py::object normalizeFunction = py::module_::import("pedalboard.midi_utils")
                                     .attr("normalize_midi_messages");

  {
    py::gil_scoped_acquire acquire;
    if (PyErr_Occurred() != nullptr)
      throw py::error_already_set();
  }

  if (normalizeFunction == py::none()) {
    throw std::runtime_error(
        "Failed to import pedalboard.midi_utils.normalize_midi_messages! This "
        "is an internal Pedalboard error and should be reported.");
  }

  py::object pyNormalizedBuffer = normalizeFunction(midiMessages);

  {
    py::gil_scoped_acquire acquire;
    if (PyErr_Occurred() != nullptr)
      throw py::error_already_set();
  }

  if (pyNormalizedBuffer == py::none()) {
    throw std::runtime_error(
        "pedalboard.midi_utils.normalize_midi_messages returned None without "
        "throwing an exception. This is an internal Pedalboard error and "
        "should be reported.");
  }

  std::vector<std::tuple<py::bytes, float>> normalizedBuffer =
      pyNormalizedBuffer.cast<std::vector<std::tuple<py::bytes, float>>>();

  for (const std::tuple<py::bytes, float> &tuple : normalizedBuffer) {
    char *messagePointer;
    py::ssize_t messageLength;

    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(std::get<0>(tuple).ptr(),
                                          &messagePointer, &messageLength)) {
      throw std::runtime_error(
          "Failed to decode Python bytes object. This is an internal "
          "Pedalboard error and should be reported.");
    }

    long sampleIndex = (std::get<1>(tuple) * sampleRate);
    buf.addEvent(messagePointer, messageLength, sampleIndex);
  }

  return buf;
}

/**
 * The VST3 and Audio Unit format managers differ in how they look up plugins
 * that are already installed on the current machine. This approach allows us to
 * return file paths from both.
 */
#if JUCE_PLUGINHOST_AU && JUCE_MAC
class AudioUnitPathFinder {
public:
  static std::vector<std::string> findInstalledAudioUnitPaths() {
    // Ensure we have a MessageManager, which is required by the VST wrapper
    // Without this, we get an assert(false) from JUCE at runtime
    juce::MessageManager::getInstance();

    juce::AudioUnitPluginFormat format;

    std::vector<std::string> pluginPaths;
    for (juce::String pluginPath : searchPathsForPlugins(
             juce::FileSearchPath(
                 "/Library/Audio/Plug-Ins/Components;~/Library/"
                 "Audio/Plug-Ins/Components"),
             true, format)) {
      pluginPaths.push_back(pluginPath.toStdString());
    }
    return pluginPaths;
  }

private:
  static juce::StringArray
  searchPathsForPlugins(const juce::FileSearchPath &directoriesToSearch,
                        const bool recursive,
                        juce::AudioUnitPluginFormat &format) {
    juce::StringArray results;

    for (int i = 0; i < directoriesToSearch.getNumPaths(); ++i) {
      recursiveFileSearch(results, directoriesToSearch[i], recursive, format);
    }

    return results;
  }

  static void recursiveFileSearch(juce::StringArray &results,
                                  const juce::File &directory,
                                  const bool recursive,
                                  juce::AudioUnitPluginFormat &format) {
    for (const auto &iter : juce::RangedDirectoryIterator(
             directory, false, "*", juce::File::findFilesAndDirectories)) {
      auto f = iter.getFile();
      bool isPlugin = false;

      if (format.fileMightContainThisPluginType(f.getFullPathName())) {
        isPlugin = true;
        results.add(f.getFullPathName());
      }

      if (recursive && (!isPlugin) && f.isDirectory())
        recursiveFileSearch(results, f, true, format);
    }
  }
};
#endif

static bool audioUnitIsInstalled(const juce::String audioUnitFilePath) {
  return !audioUnitFilePath.endsWith(".appex") &&
         !audioUnitFilePath.endsWith(".appex/") &&
         audioUnitFilePath.contains("/Library/Audio/Plug-Ins/Components/");
}

template <typename ExternalPluginType>
static juce::OwnedArray<juce::PluginDescription>
scanPluginDescriptions(std::string filename) {
  juce::MessageManager::getInstance();
  ExternalPluginType format;

  juce::OwnedArray<juce::PluginDescription> typesFound;
  std::string errorMessage = "Unable to scan plugin " + filename +
                             ": unsupported plugin format or scan failure.";

#if JUCE_PLUGINHOST_AU && JUCE_MAC
  if constexpr (std::is_same<ExternalPluginType,
                             juce::AudioUnitPluginFormat>::value) {
    auto identifiers = getAudioUnitIdentifiersFromFile(filename);
    // For each plugin in the identified bundle, scan using its AU identifier:
    for (int i = 0; i < identifiers.size(); i++) {
      std::string identifier = identifiers[i];

      juce::String pluginName, version, manufacturer;
      AudioComponentDescription componentDesc;

      bool needsAsyncInstantiation =
          juce::String(filename).endsWith(".appex") ||
          juce::String(filename).endsWith(".appex/");
      if (needsAsyncInstantiation &&
          juce::MessageManager::getInstance()->isThisTheMessageThread()) {
        // We can't scan AUv3 plugins synchronously, so we have to pump the
        // message thread and wait for the scan to complete on another thread.
        bool done = false;
        std::thread thread =
            std::thread([&identifier, &typesFound, &format, &done]() {
              format.findAllTypesForFile(typesFound, identifier);
              done = true;
            });

        // Pump the message thread until the scan is complete:
        while (!done) {
          juce::MessageManager::getInstance()->runDispatchLoopUntil(1);
        }

        thread.join();
      } else {
        format.findAllTypesForFile(typesFound, identifiers[i]);
      }
    }

    if (typesFound.isEmpty() && !audioUnitIsInstalled(filename)) {
      errorMessage += " " + AUDIO_UNIT_NOT_INSTALLED_ERROR;
    }
  } else {
    format.findAllTypesForFile(typesFound, filename);
  }
#else
  format.findAllTypesForFile(typesFound, filename);
#endif

  if (typesFound.isEmpty()) {
    throw pybind11::import_error(errorMessage);
  }

  return typesFound;
}

template <typename ExternalPluginType>
static std::vector<std::string> getPluginNamesForFile(std::string filename) {
  juce::OwnedArray<juce::PluginDescription> typesFound =
      scanPluginDescriptions<ExternalPluginType>(filename);

  std::vector<std::string> pluginNames;
  for (int i = 0; i < typesFound.size(); i++) {
    pluginNames.push_back(typesFound[i]->name.toStdString());
  }
  return pluginNames;
}

/**
 * Colours of our window frame. The host passes the colours of its own theme
 * to open_editor(); without them the frame takes the dark theme that the
 * mstand stand starts with.
 */
struct PluginWindowColours {
  /** The title bar and the frame around the editor. */
  juce::Colour bar{0xff2b2d31};
  /** The line between the title bar and the editor. */
  juce::Colour line{0xff1d1e21};
  /** The title of an active window. */
  juce::Colour text{0xffdfe0e2};
  /** Button glyphs, the title of an inactive window, a bypassed plugin. */
  juce::Colour textDim{0xff93989f};
  /** The one-pixel outline of the window. */
  juce::Colour edge{0xff4a4e54};
  /** The power button of a running plugin and the pin of a pinned window. */
  juce::Colour accent{0xff7abaff};
};

/**
 * Window colours from a dict of "#rrggbb" strings with the keys bar, line,
 * text, text_dim, edge and accent. Keys left out keep their defaults, and
 * None gives the defaults for all of them.
 */
inline PluginWindowColours parseWindowColours(py::object colors) {
  PluginWindowColours colours;
  if (colors.is_none()) {
    return colours;
  }
  if (!py::isinstance<py::dict>(colors)) {
    throw py::type_error(
        "colors must be a dict of \"#rrggbb\" strings, or None.");
  }
  static const std::map<std::string, juce::Colour PluginWindowColours::*>
      fields{{"bar", &PluginWindowColours::bar},
             {"line", &PluginWindowColours::line},
             {"text", &PluginWindowColours::text},
             {"text_dim", &PluginWindowColours::textDim},
             {"edge", &PluginWindowColours::edge},
             {"accent", &PluginWindowColours::accent}};
  for (auto item : colors.cast<py::dict>()) {
    const auto key = py::str(item.first).cast<std::string>();
    const auto field = fields.find(key);
    if (field == fields.end()) {
      throw py::key_error("Unknown window color: " + key);
    }
    const auto value = py::str(item.second).cast<std::string>();
    const bool isHex =
        value.size() == 7 && value[0] == '#' &&
        std::all_of(value.begin() + 1, value.end(), [](char c) {
          return std::isxdigit(static_cast<unsigned char>(c)) != 0;
        });
    if (!isHex) {
      throw py::value_error("Window color " + key +
                            " must look like #rrggbb, got: " + value);
    }
    colours.*(field->second) = juce::Colour((juce::uint32)(
        0xff000000u | std::stoul(value.substr(1), nullptr, 16)));
  }
  return colours;
}

/**
 * The look of our own window frame: a slim title bar in the host's colours,
 * with the power button and the title on the left and the pin and close
 * buttons on the right, and a one-pixel outline around the window.
 */
class PluginWindowLookAndFeel : public juce::LookAndFeel_V4 {
public:
  static constexpr int TITLE_BAR_HEIGHT = 22;
  static constexpr int BUTTON_WIDTH = 33;
  static constexpr int GLYPH_SIZE = 8;
  static constexpr int POWER_LEFT = 6;
  static constexpr int POWER_SIZE = 14;
  static constexpr int TITLE_GAP = 8;
  /** In pixels, as the stand sets its own text: Segoe UI 9 pt at 96 dpi. */
  static constexpr float TITLE_FONT_SIZE = 12.0f;

  PluginWindowColours colours;
  /** How many buttons stand at the right end of the title bar. */
  int rightButtons = 2;

  PluginWindowLookAndFeel()
      : juce::LookAndFeel_V4(juce::LookAndFeel_V4::getDarkColourScheme()) {}

  /** Glyphs of an inactive window fade halfway into the title bar. */
  juce::Colour glyphColour(bool activeWindow) const {
    return activeWindow ? colours.textDim
                        : colours.textDim.interpolatedWith(colours.bar, 0.5f);
  }

  /** The wash under a button in the mouse: light on a dark bar, dark on a light one. */
  juce::Colour hoverColour() const {
    return colours.bar.getPerceivedBrightness() < 0.5f
               ? juce::Colours::white.withAlpha(0.1f)
               : juce::Colours::black.withAlpha(0.08f);
  }

  void drawDocumentWindowTitleBar(juce::DocumentWindow &window,
                                  juce::Graphics &g, int w, int h,
                                  int titleSpaceX, int titleSpaceW,
                                  const juce::Image *icon,
                                  bool drawTitleTextOnLeft) override {
    juce::ignoreUnused(titleSpaceX, titleSpaceW, icon, drawTitleTextOnLeft);
    g.fillAll(colours.bar);

    const int textX = POWER_LEFT + POWER_SIZE + TITLE_GAP;
    const int textRight = w - rightButtons * BUTTON_WIDTH - TITLE_GAP;
    g.setColour(window.isActiveWindow() ? colours.text : colours.textDim);
    g.setFont(juce::Font("Segoe UI", TITLE_FONT_SIZE, juce::Font::plain)
                  .withPointHeight(TITLE_FONT_SIZE));
    g.drawText(window.getName(), textX, 0, std::max(0, textRight - textX), h,
               juce::Justification::centredLeft, true);
  }

  void drawResizableWindowBorder(juce::Graphics &g, int w, int h,
                                 const juce::BorderSize<int> &border,
                                 juce::ResizableWindow &window) override {
    juce::ignoreUnused(border, window);
    g.setColour(colours.edge);
    g.drawRect(0, 0, w, h, 1);
  }

  /** A resize border shows the same outline: JUCE would draw two translucent frames. */
  void drawResizableFrame(juce::Graphics &g, int w, int h,
                          const juce::BorderSize<int> &border) override {
    juce::ignoreUnused(border);
    g.setColour(colours.edge);
    g.drawRect(0, 0, w, h, 1);
  }
};

/** A flat button of our title bar, drawn in the colours of the window. */
class PluginTitleButton : public juce::Button {
public:
  PluginTitleButton(const juce::String &name, PluginWindowLookAndFeel &look)
      : juce::Button(name), look(look) {
    setWantsKeyboardFocus(false);
  }

protected:
  bool windowIsActive() const {
    auto *window = findParentComponentOfClass<juce::TopLevelWindow>();
    return window == nullptr || window->isActiveWindow();
  }

  /** A square of GLYPH_SIZE pixels in the middle of the button. */
  juce::Rectangle<float> glyphBox() const {
    const auto size = (float)PluginWindowLookAndFeel::GLYPH_SIZE;
    return juce::Rectangle<float>(size, size)
        .withCentre(getLocalBounds().toFloat().getCentre());
  }

  PluginWindowLookAndFeel &look;
};

/**
 * The power button: the power sign of the stand's rack, in the accent colour
 * while the plugin runs and dim while it is bypassed.
 */
class PluginPowerButton : public PluginTitleButton {
public:
  explicit PluginPowerButton(PluginWindowLookAndFeel &look)
      : PluginTitleButton("power", look) {
    setClickingTogglesState(true);
    setTooltip("Bypass");
  }

  void paintButton(juce::Graphics &g, bool highlighted, bool down) override {
    const auto colour =
        getToggleState() ? look.colours.accent : look.colours.textDim;
    g.setColour(highlighted || down ? colour.brighter(0.3f) : colour);
    // As the rack draws it: a ring of radius 5, open by 60 degrees at the
    // top, and a stroke from above the ring to its middle - centred as a whole.
    const auto centre = getLocalBounds().toFloat().getCentre();
    const float ringY = centre.y + 0.75f;
    juce::Path sign;
    sign.addCentredArc(centre.x, ringY, 5.0f, 5.0f, 0.0f,
                       juce::MathConstants<float>::pi / 6.0f,
                       juce::MathConstants<float>::pi * 11.0f / 6.0f, true);
    sign.startNewSubPath(centre.x, ringY - 6.5f);
    sign.lineTo(centre.x, ringY - 0.5f);
    g.strokePath(sign, juce::PathStrokeType(1.6f));
  }
};

/**
 * The pin. A window without it gives way to the next plugin window that
 * opens, a pinned one stays: the Target button of Pro Tools the other way
 * round. Upright in the accent colour when pinned, tilted when not.
 */
class PluginPinButton : public PluginTitleButton {
public:
  explicit PluginPinButton(PluginWindowLookAndFeel &look)
      : PluginTitleButton("pin", look) {
    setClickingTogglesState(true);
    setTooltip("Pin: keep this window when another plugin opens");
  }

  void paintButton(juce::Graphics &g, bool highlighted, bool down) override {
    if (highlighted || down) {
      g.fillAll(look.hoverColour());
    }
    // A push pin in a 12-unit square: the cap, the body widening into the
    // collar, and the needle.
    juce::Path pin;
    pin.startNewSubPath(3.5f, 1.5f);
    pin.lineTo(8.5f, 1.5f);
    pin.startNewSubPath(4.5f, 1.5f);
    pin.lineTo(4.5f, 5.5f);
    pin.lineTo(3.0f, 7.5f);
    pin.lineTo(9.0f, 7.5f);
    pin.lineTo(7.5f, 5.5f);
    pin.lineTo(7.5f, 1.5f);
    pin.startNewSubPath(6.0f, 7.5f);
    pin.lineTo(6.0f, 11.0f);
    const auto tilt =
        getToggleState()
            ? juce::AffineTransform()
            : juce::AffineTransform::rotation(
                  juce::MathConstants<float>::pi / 4.0f, 6.0f, 6.0f);
    const auto box = glyphBox();
    pin.applyTransform(tilt.scaled(box.getWidth() / 12.0f)
                           .translated(box.getX(), box.getY()));
    g.setColour(getToggleState() ? look.colours.accent
                                 : look.glyphColour(windowIsActive()));
    g.strokePath(pin, juce::PathStrokeType(1.0f));
  }
};

/** Close: a cross that turns white on red under the mouse, as in Windows. */
class PluginCloseButton : public PluginTitleButton {
public:
  explicit PluginCloseButton(PluginWindowLookAndFeel &look)
      : PluginTitleButton("close", look) {
    setTooltip("Close");
  }

  void paintButton(juce::Graphics &g, bool highlighted, bool down) override {
    const bool hot = highlighted || down;
    if (hot) {
      g.fillAll(juce::Colour(0xffe81123));
    }
    const auto box = glyphBox();
    g.setColour(hot ? juce::Colours::white
                    : look.glyphColour(windowIsActive()));
    g.drawLine(box.getX(), box.getY(), box.getRight(), box.getBottom(), 1.0f);
    g.drawLine(box.getRight(), box.getY(), box.getX(), box.getBottom(), 1.0f);
  }
};

class StandalonePluginWindow : public juce::DocumentWindow {
public:
  StandalonePluginWindow(juce::AudioProcessor &processor,
                         const juce::String &title, bool powered,
                         std::function<void(bool)> onPower,
                         std::function<void(bool)> onPin,
                         const PluginWindowColours &colours, int resizeBorder,
                         std::optional<std::pair<int, int>> position = {},
                         std::optional<std::tuple<int, int, int, int>> centreOn = {})
      : DocumentWindow(title, colours.bar, 0), processor(processor),
        onPower(std::move(onPower)), onPin(std::move(onPin)),
        resizeBorder(resizeBorder) {
    // Our own frame, drawn by JUCE, instead of the native one: the stand
    // wants its title bar with a power button and a pin, as a DAW has. The
    // buttons are ours as well, not those of DocumentWindow: there is no
    // minimise button, and the pin sits next to close.
    lookAndFeel.colours = colours;
    // Without on_pin nobody would act on the pin, so there is no pin then.
    lookAndFeel.rightButtons = this->onPin ? 2 : 1;
    setLookAndFeel(&lookAndFeel);
    setUsingNativeTitleBar(false);
    // JUCE paints the shadow of a non-native window with four extra
    // top-level windows around it; a host that watches the windows of this
    // process (or a screen reader) would see five windows instead of one.
    setDropShadowEnabled(false);
    setTitleBarHeight(PluginWindowLookAndFeel::TITLE_BAR_HEIGHT);
    setTitleBarTextCentred(false);

    powerButton.setToggleState(powered, juce::dontSendNotification);
    powerButton.onClick = [this] {
      if (this->onPower) {
        this->onPower(powerButton.getToggleState());
      }
    };
    addAndMakeVisible(powerButton);

    pinButton.onClick = [this] {
      if (this->onPin) {
        this->onPin(pinButton.getToggleState());
      }
    };
    addChildComponent(pinButton);
    pinButton.setVisible((bool)this->onPin);

    closeButton.onClick = [this] { closeButtonPressed(); };
    addAndMakeVisible(closeButton);

    // The border the host asks for lies around every editor, and the window
    // of a plugin that resizes is dragged by it. It lies around the editor and
    // not over it: the editor is a native child window and takes the mouse
    // itself. The grip sits in front of the content so that the mouse in the
    // border reaches it and not the editor's own border component.
    grip.setAlwaysOnTop(true);
    addChildComponent(grip);

    // Where the host wants the window: set before the plugin's view is
    // attached (VST3PluginWindow::componentPeerChanged) and before the window
    // is shown. JUCE creates a new window in the top-left corner of the main
    // display. Shown there, the window jumped to its place in front of the
    // user; attached there and then moved to a display with another scale,
    // the view was resized by the scale change, and FabFilter Pro-Q 4
    // remembered the smaller size. `position` is the window's top-left
    // corner; `centreOn` is a rectangle to centre the window over, within the
    // work area of that rectangle's display. Both are in physical pixels, the
    // way the host measures windows; `position` wins when both are given.
    const auto &displays = juce::Desktop::getInstance().getDisplays();
    std::optional<juce::Rectangle<int>> centreArea;
    if (position) {
      setTopLeftPosition(displays.physicalToLogical(
          juce::Point<int>(position->first, position->second)));
    } else if (centreOn) {
      const auto [x, y, width, height] = *centreOn;
      centreArea = displays.physicalToLogical(
          juce::Rectangle<int>(x, y, width, height));
      setTopLeftPosition(centreArea->getPosition());
    }

    if (processor.hasEditor()) {
      if (auto *editor = processor.createEditorIfNeeded()) {
        setContentOwned(editor, true);
        setResizable(editor->isResizable(), false);
      } else {
        throw std::runtime_error("Failed to create plugin editor UI.");
      }
    } else {
      throw std::runtime_error("Plugin has no available editor UI.");
    }

    // Centred once the view is attached: only then is its size known. The
    // window is on that display already, so its scale does not change.
    if (centreArea) {
      auto bounds = getBounds().withCentre(centreArea->getCentre());
      if (const auto *display = displays.getDisplayForRect(*centreArea)) {
        bounds = bounds.constrainedWithin(display->userArea);
      }
      setTopLeftPosition(bounds.getPosition());
    }
  }

  /**
   * One pixel all round, resizable or not. JUCE would inset a resizable
   * window by four, and at 125 % that is a five-pixel frame on three sides
   * while the top merges with the title bar.
   */
  juce::BorderSize<int> getBorderThickness() override {
    return juce::BorderSize<int>(isUsingNativeTitleBar() ? 0 : 1);
  }

  /**
   * The editor sits inside the outline, the title bar and the border. The
   * border lies around every editor, resizable or not, so that all plugin
   * windows look alike.
   */
  juce::BorderSize<int> getContentComponentBorder() override {
    auto border = DocumentWindow::getContentComponentBorder();
    border.setLeft(border.getLeft() + resizeBorder);
    border.setRight(border.getRight() + resizeBorder);
    border.setBottom(border.getBottom() + resizeBorder);
    return border;
  }

  /** Whether the window is dragged by its border: there is one, and the plugin resizes at all. */
  bool hasGrip() const { return resizeBorder > 0 && isResizable(); }

  void resized() override {
    DocumentWindow::resized();
    // The border is draggable on the left, right and bottom; the top row is
    // the outline only, so that the title bar keeps moving the window.
    grip.setBorderThickness(juce::BorderSize<int>(1, 1 + resizeBorder,
                                                  1 + resizeBorder,
                                                  1 + resizeBorder));
    grip.setBounds(getLocalBounds());
    grip.setVisible(hasGrip());
    using Look = PluginWindowLookAndFeel;
    const auto bar = getTitleBarArea();
    powerButton.setBounds(bar.getX() + Look::POWER_LEFT,
                          bar.getY() + (bar.getHeight() - Look::POWER_SIZE) / 2,
                          Look::POWER_SIZE, Look::POWER_SIZE);
    // The buttons stop short of the line under the title bar.
    const int height = bar.getHeight() - 1;
    closeButton.setBounds(bar.getRight() - Look::BUTTON_WIDTH, bar.getY(),
                          Look::BUTTON_WIDTH, height);
    pinButton.setBounds(closeButton.getX() - Look::BUTTON_WIDTH, bar.getY(),
                        Look::BUTTON_WIDTH, height);
    powerButton.toFront(false);
    pinButton.toFront(false);
    closeButton.toFront(false);
  }

  void paint(juce::Graphics &g) override {
    g.saveState();
    DocumentWindow::paint(g);
    g.restoreState();
    // The line under the title bar runs through the frame on both sides: the
    // frame of a resizable window is wider than its one-pixel outline, and
    // the title bar does not cover it.
    const auto bar = getTitleBarArea();
    g.setColour(lookAndFeel.colours.line);
    g.fillRect(1, bar.getBottom() - 1, getWidth() - 2, 1);
  }

  void activeWindowStatusChanged() override {
    DocumentWindow::activeWindowStatusChanged();
    // The glyphs fade with the window, and DocumentWindow repaints only the
    // title bar buttons of its own.
    pinButton.repaint();
    closeButton.repaint();
  }

  void setPowered(bool powered) {
    powerButton.setToggleState(powered, juce::dontSendNotification);
  }

  void setPinned(bool pinned) {
    pinButton.setToggleState(pinned, juce::dontSendNotification);
  }

  void setColours(const PluginWindowColours &colours) {
    lookAndFeel.colours = colours;
    setBackgroundColour(colours.bar);
    repaint();
  }

  /**
   * Open a native window to show a given AudioProcessor's editor UI,
   * pumping the juce::MessageManager run loop as necessary to service
   * UI events.
   *
   * Check the passed threading.Event object every 10ms to close the
   * window if necessary.
   */
  static void openWindowAndWait(juce::AudioProcessor &processor,
                                py::object optionalEvent,
                                const juce::String &title, bool powered,
                                std::function<void(bool)> onPower) {
    bool shouldThrowErrorAlreadySet = false;

    // Check the provided Event object before even opening the window:
    if (optionalEvent != py::none() &&
        optionalEvent.attr("is_set")().cast<bool>()) {
      return;
    }

    {
      // Release the GIL to allow other Python threads to run in the
      // background while we the UI is running:
      py::gil_scoped_release release;
      JUCE_AUTORELEASEPOOL {
        StandalonePluginWindow window(processor, title, powered,
                                      std::move(onPower), {},
                                      PluginWindowColours(), 0);
        window.show();

        // Run in a tight loop so that we don't have to call
        // ->stopDispatchLoop(), which causes the MessageManager to become
        // unusable in the future. The window can be closed by sending a
        // KeyboardInterrupt, closing the window in the UI, or setting the
        // provided Event object.
        while (window.isVisible()) {
          bool errorThrown = false;
          bool eventSet = false;

          {
            py::gil_scoped_acquire acquire;

            errorThrown = PyErr_CheckSignals() != 0;
            eventSet = optionalEvent != py::none() &&
                       optionalEvent.attr("is_set")().cast<bool>();
          }

          if (errorThrown || eventSet) {
            window.closeButtonPressed();
            shouldThrowErrorAlreadySet = errorThrown;
            break;
          }

          juce::MessageManager::getInstance()->runDispatchLoopUntil(10);
        }
      }

      // Once the Autorelease pool has been drained, pump the dispatch loop one
      // more time to process any window close events:
      juce::MessageManager::getInstance()->runDispatchLoopUntil(10);
    }

    if (shouldThrowErrorAlreadySet) {
      throw py::error_already_set();
    }
  }

  void closeButtonPressed() override { setVisible(false); }

  ~StandalonePluginWindow() override {
    clearContentComponent();
    setLookAndFeel(nullptr);
  }

  void show() {
    setVisible(true);
    toFront(true);
    juce::Process::makeForegroundProcess();
    // Ask again now that the window is shown: a plugin knows whether it can
    // resize only once its view is attached. Our VST3 wrapper attaches the
    // view before the window is shown; other plugin formats may not.
    if (auto *editor = dynamic_cast<juce::AudioProcessorEditor *>(
            getContentComponent())) {
      setResizable(editor->isResizable(), false);
    }
  }

private:
  juce::AudioProcessor &processor;
  std::function<void(bool)> onPower;
  std::function<void(bool)> onPin;
  /** Width of the border around the editor, in pixels; zero — none. */
  int resizeBorder;
  // The buttons draw with the look and feel, so it is constructed first.
  PluginWindowLookAndFeel lookAndFeel;
  PluginPowerButton powerButton{lookAndFeel};
  PluginPinButton pinButton{lookAndFeel};
  PluginCloseButton closeButton{lookAndFeel};
  juce::ResizableBorderComponent grip{this, getConstrainer()};

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(StandalonePluginWindow)
};

enum class ExternalPluginReloadType {
  /**
   * Unknown: we need to determine the reload type.
   */
  Unknown,

  /**
   * Most plugins are of this type: calling .reset() on them will clear their
   * internal state. This is quick and easy: to start processing a new buffer,
   * all we need to do is call .reset() and optionally prepareToPlay().
   */
  ClearsAudioOnReset,

  /**
   * This plugin type is a bit more of a pain to deal with; it could be argued
   * that plugins that don't clear their internal buffers when reset() is called
   * are buggy. To start processing a new buffer, we'll have to find another way
   * to clear the buffer, usually by reloading the plugin from scratch and
   * persisting its parameters somehow.
   */
  PersistsAudioOnReset,
};

/**
 * @brief A C++ abstract base class that gets exposed to Python as
 * `ExternalPlugin`.
 */
class AbstractExternalPlugin : public Plugin {
public:
  AbstractExternalPlugin() : Plugin() {}
};

/**
 * Make JUCE treat this process as DPI-aware, once, before it creates any
 * window.
 *
 * On Windows JUCE only sets up DPI awareness - declaring the process
 * per-monitor aware and loading the functions it needs to scale windows and
 * to tell plugins their scale factor - when it believes it is a standalone
 * application. Inside a Python process it believes it is not, so Windows
 * renders every plugin window at 96 dpi and stretches the bitmap on a
 * scaled display, and JUCE never applies the display's scale. JUCE decides
 * "standalone" by whether an application factory is registered; register
 * an empty one for the duration of the check and remove it again.
 *
 * This has to happen before the message manager exists: that creates the
 * first (hidden) window of the process, and Windows will not change the
 * DPI awareness of a process that already has windows.
 */
inline void ensureDPIAwareness() {
#if JUCE_WINDOWS
  static bool done = false;
  if (done) {
    return;
  }
  done = true;

  auto *previous = juce::JUCEApplicationBase::createInstance;
  if (previous == nullptr) {
    juce::JUCEApplicationBase::createInstance =
        []() -> juce::JUCEApplicationBase * { return nullptr; };
  }
  // Asking for the displays runs JUCE's DPI set-up.
  juce::Desktop::getInstance().getDisplays();
  juce::JUCEApplicationBase::createInstance = previous;
#endif
}

template <typename ExternalPluginType>
class ExternalPlugin : public AbstractExternalPlugin {
public:
  ExternalPlugin(
      std::string &_pathToPluginFile,
      std::optional<std::string> pluginName = {},
      float initializationTimeout = DEFAULT_INITIALIZATION_TIMEOUT_SECONDS,
      std::optional<ExternalPluginReloadType> knownReloadType = {})
      : pathToPluginFile(_pathToPluginFile),
        initializationTimeout(initializationTimeout) {
    py::gil_scoped_release release;
    // Before the first window of the process, which the message manager
    // creates: see ensureDPIAwareness().
    ensureDPIAwareness();
    // Ensure we have a MessageManager, which is required by the VST wrapper
    // Without this, we get an assert(false) from JUCE at runtime
    juce::MessageManager::getInstance();

    // When the caller already knows how this plugin behaves on reset, skip
    // the probe: detectReloadType() prepares and processes the plugin several
    // times, which costs seconds on plugins with a slow prepareToPlay().
    if (knownReloadType) {
      reloadType = *knownReloadType;
    }

    pluginFormatManager.addDefaultFormats();
    pluginFormatManager.addFormat(new juce::PatchedVST3PluginFormat());

    auto pluginFileStripped =
        pathToPluginFile.trimCharactersAtEnd(juce::File::getSeparatorString());
    auto fileExists =
        juce::File::createFileWithoutCheckingPath(pluginFileStripped).exists();

    if (!fileExists) {
      throw pybind11::import_error("Unable to load plugin " +
                                   pathToPluginFile.toStdString() +
                                   ": plugin file not found.");
    }

    juce::OwnedArray<juce::PluginDescription> typesFound =
        scanPluginDescriptions<ExternalPluginType>(
            pluginFileStripped.toStdString());

    if (!typesFound.isEmpty()) {
      if (typesFound.size() == 1) {
        foundPluginDescription = *typesFound[0];
      } else if (typesFound.size() > 1) {
        std::string errorMessage =
            "Plugin file " + pathToPluginFile.toStdString() + " contains " +
            std::to_string(typesFound.size()) + " plugins";

        // Use the provided plugin name to disambiguate:
        if (pluginName) {
          for (int i = 0; i < typesFound.size(); i++) {
            if (typesFound[i]->name.toStdString() == *pluginName) {
              foundPluginDescription = *typesFound[i];
              break;
            }
          }

          if (foundPluginDescription.name.isEmpty()) {
            errorMessage += ", and the provided plugin_name \"" + *pluginName +
                            "\" matched no plugins. ";
          }
        } else {
          errorMessage += ". ";
        }

        if (foundPluginDescription.name.isEmpty()) {
          juce::StringArray pluginNames;
          for (int i = 0; i < typesFound.size(); i++) {
            pluginNames.add(typesFound[i]->name);
          }

          errorMessage +=
              ("To open a specific plugin within this file, pass a "
               "\"plugin_name\" parameter with one of the following "
               "values:\n\t\"" +
               pluginNames.joinIntoString("\"\n\t\"").toStdString() + "\"");
          throw std::domain_error(errorMessage);
        }
      }

      reinstantiatePlugin();
    } else {
      std::string errorMessage = "Unable to load plugin " +
                                 pathToPluginFile.toStdString() +
                                 ": unsupported plugin format or load failure.";
#if JUCE_LINUX
      auto machineName = []() -> juce::String {
        struct utsname unameData;
        auto res = uname(&unameData);

        if (res != 0)
          return {};

        return unameData.machine;
      }();

      juce::File pluginBundle(pluginFileStripped);

      auto pathToSharedObjectFile =
          pluginBundle.getChildFile("Contents")
              .getChildFile(machineName + "-linux")
              .getChildFile(pluginBundle.getFileNameWithoutExtension() + ".so");

      errorMessage += " Plugin files or shared library dependencies may be "
                      "missing. (Try running `ldd \"" +
                      pathToSharedObjectFile.getFullPathName().toStdString() +
                      "\"` to see which dependencies might be missing.).";
#elif JUCE_PLUGINHOST_AU && JUCE_MAC
      if constexpr (std::is_same<ExternalPluginType,
                                 juce::AudioUnitPluginFormat>::value) {
        if (!audioUnitIsInstalled(pathToPluginFile)) {
          errorMessage += " " + AUDIO_UNIT_NOT_INSTALLED_ERROR;
        }
      }
#endif

      throw pybind11::import_error(errorMessage);
    }
  }

  ~ExternalPlugin() {
    {
      std::lock_guard<std::mutex> lock(EXTERNAL_PLUGIN_MUTEX);
      // The editor component inside the window belongs to the plugin
      // instance: the window has to go first.
      editorWindow.reset();
      if (pluginInstance) {
        pluginInstance->setPlayHead(nullptr);
      }
      pluginInstance.reset();
      NUM_ACTIVE_EXTERNAL_PLUGINS--;

      if (NUM_ACTIVE_EXTERNAL_PLUGINS == 0) {
        juce::DeletedAtShutdown::deleteAll();
        juce::MessageManager::deleteInstance();
      }
    }
  }

  struct SetPresetVisitor : public juce::ExtensionsVisitor {
    const juce::MemoryBlock &presetData;
    bool didSetPreset;

    SetPresetVisitor(const juce::MemoryBlock &presetData)
        : presetData(presetData), didSetPreset(false) {}

    void visitVST3Client(
        const juce::ExtensionsVisitor::VST3Client &client) override {
      this->didSetPreset = client.setPreset(presetData);
    }
  };

  void loadPresetFile(std::string presetFilePath) {
    juce::File presetFile(presetFilePath);
    juce::MemoryBlock presetData;

    if (!presetFile.loadFileAsData(presetData)) {
      throw std::runtime_error("Failed to read preset file: " + presetFilePath);
    }

    SetPresetVisitor visitor{presetData};
    pluginInstance->getExtensions(visitor);
    if (!visitor.didSetPreset) {
      throw std::runtime_error("Plugin failed to load data from preset file: " +
                               presetFilePath);
    }
  }

  void setPreset(const void *data, size_t size) {
    juce::MemoryBlock presetData(data, size);
    SetPresetVisitor visitor{presetData};
    pluginInstance->getExtensions(visitor);
    if (!visitor.didSetPreset) {
      throw std::runtime_error("Failed to set preset data for plugin: " +
                               pathToPluginFile.toStdString());
    }
  }

  struct GetPresetVisitor : public juce::ExtensionsVisitor {
    // This block will get updated with the current preset data when
    // visiting VST3 clients.
    juce::MemoryBlock &presetData;
    bool didGetPreset;

    GetPresetVisitor(juce::MemoryBlock &presetData)
        : presetData(presetData), didGetPreset(false) {}

    void visitVST3Client(
        const juce::ExtensionsVisitor::VST3Client &client) override {
      this->presetData = client.getPreset();
      this->didGetPreset = true;
    }
  };

  void getPreset(juce::MemoryBlock &dest) const {
    // Get the plugin state's .vstpreset representation if possible.
    GetPresetVisitor visitor(dest);
    pluginInstance->getExtensions(visitor);

    if (!visitor.didGetPreset) {
      throw std::runtime_error("Failed to get preset data for plugin " +
                               pathToPluginFile.toStdString());
    }
  }

  void reinstantiatePlugin() {
    // JUCE only allows creating new plugin instances from the main
    // thread, which we may not be on:
    if (!juce::MessageManager::getInstance()->isThisTheMessageThread()) {
      throw std::runtime_error(
          "Plugin " + pathToPluginFile.toStdString() +
          " must be reloaded on the main thread. Please pass `reset=False` " +
          "if calling this plugin from a non-main thread.");
    }

    // If we have an existing plugin, save its state and reload its state
    // later:
    juce::MemoryBlock savedState;
    std::map<int, float> currentParameters;
    const bool replacingInstance = pluginInstance != nullptr;

    if (pluginInstance) {
      pluginInstance->getStateInformation(savedState);

      for (auto *parameter : pluginInstance->getParameters()) {
        currentParameters[parameter->getParameterIndex()] =
            parameter->getValue();
      }

      {
        std::lock_guard<std::mutex> lock(EXTERNAL_PLUGIN_MUTEX);
        // Delete the plugin instance itself (and its editor window first):
        editorWindow.reset();
        pluginInstance->setPlayHead(nullptr);
        pluginInstance.reset();
        NUM_ACTIVE_EXTERNAL_PLUGINS--;
      }
    }

    juce::String loadError;
    {
      std::lock_guard<std::mutex> lock(EXTERNAL_PLUGIN_MUTEX);

      pluginInstance =
          createPluginInstance(foundPluginDescription, ExternalLoadSampleRate,
                               ExternalLoadMaximumBlockSize, loadError);

      if (!pluginInstance) {
        throw pybind11::import_error("Unable to load plugin " +
                                     pathToPluginFile.toStdString() + ": " +
                                     loadError.toStdString());
      }

      pluginInstance->setPlayHead(&playHead);
      pluginInstance->enableAllBuses();

      auto mainInputBus = pluginInstance->getBus(true, 0);
      auto mainOutputBus = pluginInstance->getBus(false, 0);

      if (!mainOutputBus) {
        auto exception = std::invalid_argument(
            "Plugin '" + pluginInstance->getName().toStdString() +
            "' does not produce audio output.");
        pluginInstance.reset();
        throw exception;
      }

      if (reloadType == ExternalPluginReloadType::Unknown) {
        reloadType = detectReloadType();
        if (reloadType == ExternalPluginReloadType::PersistsAudioOnReset) {
          // Reload again, as we just passed audio into a plugin that
          // we know doesn't reset itself cleanly!
          pluginInstance = createPluginInstance(
              foundPluginDescription, ExternalLoadSampleRate,
              ExternalLoadMaximumBlockSize, loadError);

          if (!pluginInstance) {
            throw pybind11::import_error("Unable to load plugin " +
                                         pathToPluginFile.toStdString() + ": " +
                                         loadError.toStdString());
          }
          pluginInstance->setPlayHead(&playHead);
        }
      }

      NUM_ACTIVE_EXTERNAL_PLUGINS++;
    }

    pluginInstance->setStateInformation(savedState.getData(),
                                        savedState.getSize());

    // Set all of the parameters twice: we may have meta-parameters that
    // change the validity of other `setValue` calls. (i.e.: param1 can't be
    // set until param2 is set.)
    for (int i = 0; i < 2; i++) {
      for (auto *parameter : pluginInstance->getParameters()) {
        if (currentParameters.count(parameter->getParameterIndex()) > 0) {
          parameter->setValue(
              currentParameters[parameter->getParameterIndex()]);
        }
      }
    }

    // The new instance is not prepared yet. Prepare it the way the previous
    // one was, if it was; otherwise the next block does, channel layout
    // included.
    const juce::dsp::ProcessSpec previousSpec = preparedSpec;
    preparedSpec = {0};
    if (previousSpec.maximumBlockSize != 0 && previousSpec.numChannels != 0) {
      prepare(previousSpec);
      // A plugin reloaded on reset with its saved state is switched off and
      // on once after being prepared, as before: MagicDeathEyeStereo renders
      // differently on every reset without it (measured). The instance
      // stays active, so no further prepare follows.
      if (replacingInstance) {
        pluginInstance->reset();
      }
    }

    // A freshly created instance is not reset: JUCE's reset() switches the
    // component on without marking it active, so the next prepare() woke
    // the plugin a second time - and a wake costs seconds on some plugins
    // (AL-1: 0.9 s). Latency compensation starts over with the new
    // instance, also after the reload-type probe.
    samplesProvided = 0;

    // Try to warm up the plugin.
    // Some plugins (mostly instrument plugins) may load resources on start;
    // this call attempts to give them time to load those resources.
    attemptToWarmUp();
  }

  void setNumChannels(int numChannels) {
    if (!pluginInstance)
      return;

    if (numChannels == 0)
      return;

    auto mainInputBus = pluginInstance->getBus(true, 0);
    auto mainOutputBus = pluginInstance->getBus(false, 0);

    // Try to disable all non-main input buses if possible:
    for (int i = 1; i < pluginInstance->getBusCount(true); i++) {
      auto *bus = pluginInstance->getBus(true, i);
      if (bus->isNumberOfChannelsSupported(0))
        bus->enable(false);
    }

    // ...and all non-main output buses too:
    for (int i = 1; i < pluginInstance->getBusCount(false); i++) {
      auto *bus = pluginInstance->getBus(false, i);
      if (bus->isNumberOfChannelsSupported(0))
        bus->enable(false);
    }

    if ((!mainInputBus || mainInputBus->getNumberOfChannels() == numChannels) &&
        mainOutputBus->getNumberOfChannels() == numChannels) {
      return;
    }

    // Cache these values in case the plugin fails to update:
    auto previousInputChannelCount =
        mainInputBus ? mainInputBus->getNumberOfChannels() : 0;
    auto previousOutputChannelCount = mainOutputBus->getNumberOfChannels();

    // Try to change the input and output bus channel counts...
    if (mainInputBus)
      mainInputBus->setNumberOfChannels(numChannels);
    mainOutputBus->setNumberOfChannels(numChannels);

    // If, post-reload, we still can't use the right number of channels, let's
    // conclude the plugin doesn't allow this channel count.
    if ((!mainInputBus || mainInputBus->getNumberOfChannels() != numChannels) ||
        mainOutputBus->getNumberOfChannels() != numChannels) {

      // Reset the bus configuration to what it was before, so we don't
      // leave one of the buses smaller than the other:
      if (mainInputBus)
        mainInputBus->setNumberOfChannels(previousInputChannelCount);
      mainOutputBus->setNumberOfChannels(previousOutputChannelCount);

      throw std::invalid_argument(
          "Plugin '" + pluginInstance->getName().toStdString() +
          "' does not support " + std::to_string(numChannels) +
          "-channel output. (Main bus currently expects " +
          std::to_string(mainInputBus ? mainInputBus->getNumberOfChannels()
                                      : 0) +
          " input channels and " +
          std::to_string(mainOutputBus->getNumberOfChannels()) +
          " output channels.)");
    }
  }

  const juce::String getName() const {
    return pluginInstance ? pluginInstance->getName() : "<unknown>";
  }

  const int getNumChannels() const {
    // Input and output channel counts should match.
    if (!pluginInstance) {
      return 0;
    }

    auto mainInputBus = pluginInstance->getBus(true, 0);
    if (!mainInputBus) {
      return 0;
    }

    return mainInputBus->getNumberOfChannels();
  }

  /**
   * Send a MIDI note into this plugin in an attempt to wait for the plugin to
   * "warm up". Many plugins do asynchronous background tasks on launch (such as
   * loading assets from disk, etc). These background tasks may depend on the
   * event loop, which Pedalboard does not pump by default.
   *
   * Returns true if the plugin rendered audio within the allotted timeout;
   * false if no audio was received before the timeout expired.
   */
  bool attemptToWarmUp() {
    if (!pluginInstance || initializationTimeout <= 0)
      return false;

    auto endTime = juce::Time::currentTimeMillis() +
                   (long)(initializationTimeout * 1000.0);

    const int numInputChannels = pluginInstance->getMainBusNumInputChannels();
    const float sampleRate = 44100.0f;
    const int bufferSize = 2048;

    if (numInputChannels != 0) {
      // TODO: For effect plugins, do this check as well!
      return false;
    }

    // Set input and output buses/channels appropriately:
    int numOutputChannels =
        std::max(pluginInstance->getMainBusNumInputChannels(),
                 pluginInstance->getMainBusNumOutputChannels());
    setNumChannels(numOutputChannels);
    pluginInstance->setNonRealtime(true);
    pluginInstance->prepareToPlay(sampleRate, bufferSize);

    // Prepare an empty MIDI buffer to measure the background noise of the
    // plugin:
    juce::MidiBuffer emptyNoteBuffer;

    // Send in a MIDI buffer containing a single middle C at full velocity:
    auto noteOn = juce::MidiMessage::noteOn(
        /* channel */ 1, /* note number */ 60, /* velocity */ (juce::uint8)127);

    // And prepare an all-notes-off buffer:
    auto allNotesOff = juce::MidiMessage::allNotesOff(/* channel */ 1);

    if (juce::MessageManager::getInstance()->isThisTheMessageThread()) {
      for (int i = 0; i < 10; i++) {
        if (juce::Time::currentTimeMillis() >= endTime)
          return false;
        juce::MessageManager::getInstance()->runDispatchLoopUntil(1);
      }
    }

    juce::AudioBuffer<float> audioBuffer(numOutputChannels, bufferSize);
    audioBuffer.clear();

    pluginInstance->processBlock(audioBuffer, emptyNoteBuffer);
    auto noiseFloor = audioBuffer.getMagnitude(0, bufferSize);

    audioBuffer.clear();

    // Now pass in a middle C:
    // Note: we create a new MidiBuffer every time here, as unlike AudioBuffer,
    // the messages in a MidiBuffer get erased every time we call processBlock!
    {
      juce::MidiBuffer noteOnBuffer(noteOn);
      pluginInstance->processBlock(audioBuffer, noteOnBuffer);
    }

    // Then keep pumping the message thread until we get some louder output:
    bool magnitudeIncreased = false;
    while (true) {
      auto magnitudeWithNoteHeld = audioBuffer.getMagnitude(0, bufferSize);
      if (magnitudeWithNoteHeld > noiseFloor * 5) {
        magnitudeIncreased = true;
        break;
      }

      if (juce::MessageManager::getInstance()->isThisTheMessageThread()) {
        for (int i = 0; i < 10; i++) {
          juce::MessageManager::getInstance()->runDispatchLoopUntil(1);
        }
      }

      if (juce::Time::currentTimeMillis() >= endTime)
        break;

      audioBuffer.clear();
      {
        juce::MidiBuffer noteOnBuffer(noteOn);
        pluginInstance->processBlock(audioBuffer, noteOnBuffer);
      }

      if (juce::Time::currentTimeMillis() >= endTime)
        break;
    }

    // Send in an All Notes Off and then reset, just to make sure we clear any
    // note trails:
    audioBuffer.clear();
    {
      juce::MidiBuffer allNotesOffBuffer(allNotesOff);
      pluginInstance->processBlock(audioBuffer, allNotesOffBuffer);
    }
    pluginInstance->reset();
    pluginInstance->releaseResources();

    return magnitudeIncreased;
  }

  /**
   * Send some audio through the plugin to detect if the ->reset() call
   * actually resets internal buffers. This determines how quickly we
   * can reset the plugin and is only called on instantiation.
   */
  ExternalPluginReloadType detectReloadType() {
    const int numInputChannels = pluginInstance->getMainBusNumInputChannels();
    const int bufferSize = 512;
    const float sampleRate = 44100.0f;

    if (numInputChannels == 0) {
      // TODO: For instrument plugins, figure out how to measure audio
      // persistence across resets.
      return ExternalPluginReloadType::Unknown;
    }

    // Set input and output buses/channels appropriately:
    setNumChannels(numInputChannels);
    pluginInstance->setNonRealtime(true);
    pluginInstance->prepareToPlay(sampleRate, bufferSize);

    // Send in a buffer full of silence to get a baseline noise level:
    juce::AudioBuffer<float> audioBuffer(numInputChannels, bufferSize);
    juce::MidiBuffer emptyMidiBuffer;

    // Process the silent buffer a couple of times to give the plugin time to
    // "warm up"
    for (int i = 0; i < 5; i++) {
      audioBuffer.clear();
      {
        juce::dsp::AudioBlock<float> block(audioBuffer);
        juce::dsp::ProcessContextReplacing<float> context(block);
        process(context);
      }
    }

    // Measure the noise floor of the plugin:
    auto noiseFloor = audioBuffer.getMagnitude(0, bufferSize);

    // Reset:
    pluginInstance->releaseResources();
    pluginInstance->setNonRealtime(true);
    pluginInstance->prepareToPlay(sampleRate, bufferSize);

    juce::Random random;

    // Send noise into the plugin:
    for (int i = 0; i < 5; i++) {
      for (auto i = 0; i < bufferSize; i++) {
        for (int c = 0; c < numInputChannels; c++) {
          audioBuffer.setSample(c, i, (random.nextFloat() * 2.0f) - 1.0f);
        }
      }
      juce::dsp::AudioBlock<float> block(audioBuffer);
      juce::dsp::ProcessContextReplacing<float> context(block);
      process(context);
    }

    auto signalVolume = audioBuffer.getMagnitude(0, bufferSize);

    // Reset again, and send in silence:
    pluginInstance->releaseResources();
    pluginInstance->setNonRealtime(true);
    pluginInstance->prepareToPlay(sampleRate, bufferSize);
    audioBuffer.clear();
    {
      juce::dsp::AudioBlock<float> block(audioBuffer);
      juce::dsp::ProcessContextReplacing<float> context(block);
      process(context);
    }

    auto magnitudeOfSilentBuffer = audioBuffer.getMagnitude(0, bufferSize);

    // If the silent buffer we passed in post-reset is noticeably louder
    // than the first buffer we passed in, this plugin probably persists
    // internal state across calls to releaseResources().
    bool pluginPersistsAudioOnReset = magnitudeOfSilentBuffer > noiseFloor * 5;

    if (pluginPersistsAudioOnReset) {
      return ExternalPluginReloadType::PersistsAudioOnReset;
    } else {
      return ExternalPluginReloadType::ClearsAudioOnReset;
    }
  }

  /**
   * reset() is only called if reset=True is passed.
   */
  void reset() override {
    if (pluginInstance) {
      switch (reloadType) {
      case ExternalPluginReloadType::ClearsAudioOnReset:
        // Deactivate and prepare again right away with the same spec: the
        // very cycle detectReloadType() classified this plugin with. One
        // wake, and the plugin stays ready for the next block. (JUCE's own
        // reset() switched the plugin off and on, and the releaseResources()
        // after it made the next block wake the plugin a second time.)
        if (preparedSpec.maximumBlockSize != 0) {
          const juce::dsp::ProcessSpec spec = preparedSpec;
          pluginInstance->releaseResources();
          preparedSpec.maximumBlockSize = 0;
          prepare(spec);
        }
        break;

      case ExternalPluginReloadType::Unknown:
      case ExternalPluginReloadType::PersistsAudioOnReset:
        // reinstantiatePlugin() prepares the new instance with the last
        // spec when there was one.
        pluginInstance->releaseResources();
        reinstantiatePlugin();
        break;
      default:
        throw std::runtime_error("Plugin reload type is an invalid value (" +
                                 std::to_string((int)reloadType) +
                                 ") - this likely indicates a programming "
                                 "error or memory corruption.");
      }

      samplesProvided = 0;
    }
  }

  /**
   * prepare() is called on every render call, regardless of if the plugin has
   * been reset.
   */
  void prepare(const juce::dsp::ProcessSpec &spec) override {
    if (!pluginInstance) {
      return;
    }

    // Decided by preparedSpec - what the instance is actually prepared
    // with - and not by lastSpec: Python's reset() forgets lastSpec
    // (resetLastChannelLayout) while the instance stays prepared, and going
    // by it woke the plugin a second time after every reset.
    lastSpec = spec;
    if (!isPreparedFor(spec)) {

      // Changing the number of channels requires releaseResources to be
      // called:
      if (preparedSpec.numChannels != spec.numChannels) {
        pluginInstance->releaseResources();
        setNumChannels(spec.numChannels);
      }

      // Offline unless the caller asked for real time (see ``realtime``):
      // plugins may pick different algorithms, and some refuse to behave like
      // they do during DAW playback when told they are being rendered.
      pluginInstance->setNonRealtime(!realtime);
      playHead.sampleRate = spec.sampleRate;
      pluginInstance->prepareToPlay(spec.sampleRate, spec.maximumBlockSize);

      preparedSpec = spec;
    }
  }

  int process(
      const juce::dsp::ProcessContextReplacing<float> &context) override {

    if (pluginInstance) {
      juce::MidiBuffer emptyMidiBuffer;

      if (pluginInstance->getMainBusNumInputChannels() == 0 &&
          context.getInputBlock().getNumChannels() > 0) {
        throw std::invalid_argument(
            "Plugin '" + pluginInstance->getName().toStdString() +
            "' does not accept audio input. It may be an instrument plugin "
            "instead of an effect plugin.");
      }

      const juce::dsp::AudioBlock<const float> &inputBlock =
          context.getInputBlock();
      juce::dsp::AudioBlock<float> &outputBlock = context.getOutputBlock();

      if ((size_t)pluginInstance->getMainBusNumInputChannels() !=
          inputBlock.getNumChannels()) {
        throw std::invalid_argument(
            "Plugin '" + pluginInstance->getName().toStdString() +
            "' was instantiated with " +
            std::to_string(pluginInstance->getMainBusNumInputChannels()) +
            "-channel input, but provided audio data contained " +
            std::to_string(inputBlock.getNumChannels()) + " channel" +
            (inputBlock.getNumChannels() == 1 ? "" : "s") + ".");
      }

      if ((size_t)pluginInstance->getMainBusNumOutputChannels() <
          inputBlock.getNumChannels()) {
        throw std::invalid_argument(
            "Plugin '" + pluginInstance->getName().toStdString() +
            "' produces " +
            std::to_string(pluginInstance->getMainBusNumOutputChannels()) +
            "-channel output, but provided audio data contained " +
            std::to_string(inputBlock.getNumChannels()) + " channel" +
            (inputBlock.getNumChannels() == 1 ? " " : "s") +
            ". (The number of channels returned must match the "
            "number of channels passed in.)");
      }

      // A plugin may keep more input buses enabled than we feed it: a
      // sidechain it refuses to disable in setNumChannels(), for example.
      // JUCE fills any channel that the buffer does not cover with a null
      // pointer, and a plugin that reads its sidechain unconditionally then
      // crashes. So size the buffer for every enabled input *and* output
      // channel; the extra channels are silent dummies (see below).
      std::vector<float *> channelPointers(
          std::max(pluginInstance->getTotalNumInputChannels(),
                   pluginInstance->getTotalNumOutputChannels()));

      for (size_t i = 0; i < outputBlock.getNumChannels(); i++) {
        channelPointers[i] = outputBlock.getChannelPointer(i);
      }

      // Depending on the bus layout, we may have to pass extra buffers to the
      // plugin that we don't use. Use vector here to ensure the memory is
      // freed via RAII.
      std::vector<std::vector<float>> dummyChannels;
      for (size_t i = outputBlock.getNumChannels(); i < channelPointers.size();
           i++) {
        std::vector<float> dummyChannel(outputBlock.getNumSamples());
        channelPointers[i] = dummyChannel.data();
        dummyChannels.push_back(std::move(dummyChannel));
      }

      // Create an audio buffer that doesn't actually allocate anything, but
      // just points to the data in the ProcessContext.
      juce::AudioBuffer<float> audioBuffer(channelPointers.data(),
                                           channelPointers.size(),
                                           outputBlock.getNumSamples());

      pluginInstance->processBlock(audioBuffer, emptyMidiBuffer);
      samplesProvided += outputBlock.getNumSamples();
      if (playHead.playing) {
        playHead.positionSamples += (juce::int64)outputBlock.getNumSamples();
      }

      // To compensate for any latency added by the plugin,
      // only tell Pedalboard to use the last _n_ samples.
      long usableSamplesProduced =
          std::max(0L, samplesProvided - pluginInstance->getLatencySamples());
      return static_cast<int>(
          std::min(usableSamplesProduced, (long)outputBlock.getNumSamples()));
    }

    return 0;
  }

  py::array_t<float> renderMIDIMessages(py::object midiMessages, float duration,
                                        float sampleRate,
                                        unsigned int numChannels,
                                        unsigned long bufferSize, bool reset) {

    // Tiny quality-of-life improvement to try to detect if people have swapped
    // the duration and sample_rate arguments:
    if ((duration == 48000 || duration == 44100 || duration == 22050 ||
         duration == 11025) &&
        sampleRate < 8000) {
      throw std::invalid_argument(
          "Plugin '" + pluginInstance->getName().toStdString() +
          "' was called with a duration argument of " +
          std::to_string(duration) + " and a sample_rate argument of " +
          std::to_string(sampleRate) +
          ". These arguments appear to be flipped, and may cause distorted "
          "audio to be rendered. Try reversing the order of the sample_rate "
          "and duration arguments provided to this method.");
    }

    std::scoped_lock<std::mutex>(this->mutex);

    py::array_t<float> outputArray;
    unsigned long outputSampleCount = duration * sampleRate;

    juce::MidiBuffer midiInputBuffer =
        parseMidiBufferFromPython(midiMessages, sampleRate);

    outputArray =
        py::array_t<float>({numChannels, (unsigned int)outputSampleCount});

    float *outputArrayPointer = static_cast<float *>(outputArray.request().ptr);

    py::gil_scoped_release release;

    if (pluginInstance) {
      if (reset)
        this->reset();

      juce::dsp::ProcessSpec spec;
      spec.sampleRate = sampleRate;
      spec.maximumBlockSize = (juce::uint32)bufferSize;
      spec.numChannels = (juce::uint32)numChannels;
      prepare(spec);

      if (!foundPluginDescription.isInstrument) {
        throw std::invalid_argument(
            "Plugin '" + pluginInstance->getName().toStdString() +
            "' expects audio as input, but was provided MIDI messages.");
      }

      if ((size_t)pluginInstance->getMainBusNumOutputChannels() !=
          numChannels) {
        throw std::invalid_argument(
            "Plugin '" + pluginInstance->getName().toStdString() +
            "' produces " +
            std::to_string(pluginInstance->getMainBusNumOutputChannels()) +
            "-channel output, but " + std::to_string(numChannels) +
            " channels of output were requested.");
      }

      std::memset((void *)outputArrayPointer, 0,
                  sizeof(float) * numChannels * outputSampleCount);

      for (unsigned long i = 0; i < outputSampleCount; i += bufferSize) {
        unsigned long chunkSampleCount =
            std::min((unsigned long)bufferSize, outputSampleCount - i);

        std::vector<float *> channelPointers(numChannels);
        for (size_t c = 0; c < numChannels; c++) {
          channelPointers[c] =
              (outputArrayPointer + (outputSampleCount * c) + i);
        }

        // Create an audio buffer that doesn't actually allocate anything, but
        // just points to the data in the output array.
        juce::AudioBuffer<float> audioChunk(
            channelPointers.data(), channelPointers.size(), chunkSampleCount);

        juce::MidiBuffer midiChunk;
        midiChunk.addEvents(midiInputBuffer, i, chunkSampleCount, -i);

        pluginInstance->processBlock(audioChunk, midiChunk);
      }
    }

    return outputArray;
  }

  void getState(juce::MemoryBlock &dest) const {
    pluginInstance->getStateInformation(dest);
  }

  void setState(const void *data, size_t size) {
    pluginInstance->setStateInformation(data, size);
  }

  std::vector<juce::AudioProcessorParameter *> getParameters() const {
    std::vector<juce::AudioProcessorParameter *> parameters;
    for (auto *parameter : pluginInstance->getParameters()) {
      parameters.push_back(parameter);
    }
    return parameters;
  }

  juce::AudioProcessorParameter *getParameter(const std::string &name) const {
    for (auto *parameter : pluginInstance->getParameters()) {
      if (parameter->getName(512).toStdString() == name) {
        return parameter;
      }
    }
    return nullptr;
  }

  virtual int getLatencyHint() override {
    if (!pluginInstance)
      return 0;
    return pluginInstance->getLatencySamples();
  }

  virtual bool acceptsAudioInput() override {
    return pluginInstance && pluginInstance->getMainBusNumInputChannels() > 0;
  }

  /**
   * A plain function for the editor window that calls the Python callable
   * kept in `holder`. The callable lives in a member so that it is copied and
   * released while we hold the GIL; the function acquires the GIL when a
   * button is clicked, and prints an exception instead of letting it unwind
   * through JUCE's event loop.
   */
  std::function<void(bool)> windowCallback(py::object &holder) {
    if (holder.is_none()) {
      return {};
    }
    return [&holder](bool state) {
      py::gil_scoped_acquire acquire;
      try {
        holder(state);
      } catch (py::error_already_set &e) {
        e.restore();
        PyErr_Print();
      }
    };
  }

  void showEditor(py::object optionalEvent, std::optional<std::string> title,
                  bool powered, py::object onPower) {
    if (!pluginInstance) {
      throw std::runtime_error(
          "Editor cannot be shown - plugin not loaded. This is an internal "
          "Pedalboard error and should be reported.");
    }

    if (onPower != py::none() && !py::hasattr(onPower, "__call__")) {
      throw py::type_error("on_power must be callable or None.");
    }

    if (optionalEvent != py::none() && !py::hasattr(optionalEvent, "is_set")) {
      throw py::type_error(
          "Pedalboard expected a threading.Event object to be "
          "passed to show_editor, but the provided object (\"" +
          py::repr(optionalEvent).cast<std::string>() +
          "\") does not have an 'is_set' method.");
    }

    {
      py::gil_scoped_release release;
      if (!juce::Desktop::getInstance().getDisplays().getPrimaryDisplay()) {
        throw std::runtime_error(
            "Editor cannot be shown - no visual display devices available.");
      }

      if (!juce::MessageManager::getInstance()->isThisTheMessageThread()) {
        throw std::runtime_error(
            "Plugin UI windows can only be shown from the main thread.");
      }
    }

    editorPowerCallback = onPower;
    const auto callback = windowCallback(editorPowerCallback);

    const juce::String windowTitle =
        title ? juce::String(*title)
              : juce::String(pluginInstance->getName());
    try {
      StandalonePluginWindow::openWindowAndWait(
          *pluginInstance, optionalEvent, windowTitle, powered, callback);
    } catch (...) {
      editorPowerCallback = py::none();
      throw;
    }
    editorPowerCallback = py::none();
  }

  /**
   * Open the editor window and return at once. The caller owns the event
   * loop from then on: it has to call pumpEditors() regularly on this same
   * (message) thread, or the window will not repaint or react.
   */
  void openEditor(std::optional<std::string> title, bool powered,
                  py::object onPower, py::object onPin, py::object colors,
                  int resizeBorder,
                  std::optional<std::pair<int, int>> position,
                  std::optional<std::tuple<int, int, int, int>> centreOn) {
    if (!pluginInstance) {
      throw std::runtime_error(
          "Editor cannot be shown - plugin not loaded. This is an internal "
          "Pedalboard error and should be reported.");
    }
    if (onPower != py::none() && !py::hasattr(onPower, "__call__")) {
      throw py::type_error("on_power must be callable or None.");
    }
    if (onPin != py::none() && !py::hasattr(onPin, "__call__")) {
      throw py::type_error("on_pin must be callable or None.");
    }
    if (resizeBorder < 0) {
      throw py::value_error("resize_border must be 0 or more.");
    }
    const PluginWindowColours colours = parseWindowColours(colors);
    checkEditorThread();

    if (editorWindow) {
      if (editorWindow->isVisible()) {
        py::gil_scoped_release release;
        editorWindow->toFront(true);
        return;
      }
      editorWindow.reset();
    }

    editorPowerCallback = onPower;
    editorPinCallback = onPin;
    auto powerCallback = windowCallback(editorPowerCallback);
    auto pinCallback = windowCallback(editorPinCallback);
    const juce::String windowTitle =
        title ? juce::String(*title)
              : juce::String(pluginInstance->getName());

    py::gil_scoped_release release;
    editorWindow = std::make_unique<StandalonePluginWindow>(
        *pluginInstance, windowTitle, powered, std::move(powerCallback),
        std::move(pinCallback), colours, resizeBorder, position, centreOn);
    editorWindow->show();
    juce::MessageManager::getInstance()->runDispatchLoopUntil(1);
  }

  /** Close the editor window opened by openEditor(), if any. */
  void closeEditor() {
    checkEditorThread();
    {
      py::gil_scoped_release release;
      if (editorWindow) {
        editorWindow->setVisible(false);
        editorWindow.reset();
        juce::MessageManager::getInstance()->runDispatchLoopUntil(1);
      }
    }
    editorPowerCallback = py::none();
    editorPinCallback = py::none();
  }

  /**
   * Whether the window opened by openEditor() is still on screen. A window
   * that the user closed is released here, on the message thread.
   */
  bool isEditorOpen() {
    if (editorWindow && !editorWindow->isVisible()) {
      closeEditor();
    }
    return editorWindow != nullptr;
  }

  /** The native handle (HWND on Windows) of the open editor window, or 0. */
  std::uintptr_t editorHandle() {
    if (!isEditorOpen()) {
      return 0;
    }
    if (auto *peer = editorWindow->getPeer()) {
      return (std::uintptr_t)peer->getNativeHandle();
    }
    return 0;
  }

  /** Reflect a bypass change made elsewhere on the window's power button. */
  void setEditorPowered(bool powered) {
    if (isEditorOpen()) {
      py::gil_scoped_release release;
      editorWindow->setPowered(powered);
    }
  }

  /** Set the pin of the open editor window from outside, without on_pin. */
  void setEditorPinned(bool pinned) {
    if (isEditorOpen()) {
      py::gil_scoped_release release;
      editorWindow->setPinned(pinned);
    }
  }

  /** Repaint the frame of the open editor window in new colours. */
  void setEditorColours(py::object colors) {
    const PluginWindowColours colours = parseWindowColours(colors);
    if (isEditorOpen()) {
      py::gil_scoped_release release;
      editorWindow->setColours(colours);
    }
  }

  /**
   * Run the event loop for the windows opened by openEditor() for the given
   * time. Call this from the message thread, between other work.
   */
  static void pumpEditors(int milliseconds) {
    checkEditorThread();
    py::gil_scoped_release release;
    juce::MessageManager::getInstance()->runDispatchLoopUntil(milliseconds);
  }

  static void checkEditorThread() {
    py::gil_scoped_release release;
    if (!juce::Desktop::getInstance().getDisplays().getPrimaryDisplay()) {
      throw std::runtime_error(
          "Editor cannot be shown - no visual display devices available.");
    }
    if (!juce::MessageManager::getInstance()->isThisTheMessageThread()) {
      throw std::runtime_error(
          "Plugin UI windows can only be shown from the main thread.");
    }
  }

  ExternalPluginReloadType reloadType = ExternalPluginReloadType::Unknown;
  juce::PluginDescription foundPluginDescription;

  /**
   * Whether the plugin is told that it runs in real time, as during playback
   * in a DAW, rather than being rendered offline. Off by default, which is
   * what Pedalboard has always done.
   */
  bool realtime = false;

  /**
   * The spec the plugin instance is prepared with right now; all zeros when
   * it is not prepared. Unlike lastSpec, it is not forgotten by Python's
   * reset(), and it is what prepare() goes by.
   */
  juce::dsp::ProcessSpec preparedSpec = {0};

  void setRealtime(bool value) {
    if (realtime == value) {
      return;
    }
    realtime = value;
    // JUCE hands the mode to a VST3 plugin in setupProcessing(), i.e. from
    // prepareToPlay() - and skips prepareToPlay() altogether on an instance
    // that is still active with the same sample rate and block size. So an
    // active plugin is released here, the way a DAW deactivates plugins
    // around an offline render, and the next prepare() runs prepareToPlay()
    // in full with the new mode. Like reset(), this starts the plugin's
    // internal state and its latency buffer over: switch before feeding
    // audio, or call prepare() afterwards.
    if (pluginInstance && preparedSpec.maximumBlockSize != 0) {
      pluginInstance->releaseResources();
      samplesProvided = 0;
    }
    preparedSpec.maximumBlockSize = 0;
  }

  /**
   * Wake the plugin now for blocks of up to maximumBlockSize samples at
   * sampleRate with numChannels channels, without passing audio through it:
   * what a DAW does when a plugin is inserted. The first block of that size
   * then costs nothing extra. A plugin already prepared for a different
   * configuration is deactivated first - the VST3 spec requires that before
   * setupProcessing(), and JUCE's prepareToPlay() does not do it - and its
   * latency compensation starts over.
   */
  void prepareForPlayback(double sampleRate, unsigned int maximumBlockSize,
                          unsigned int numChannels) {
    if (!pluginInstance) {
      return;
    }
    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = static_cast<juce::uint32>(maximumBlockSize);
    spec.numChannels = static_cast<juce::uint32>(numChannels);

    if (!isPreparedFor(spec) && preparedSpec.maximumBlockSize != 0) {
      pluginInstance->releaseResources();
      samplesProvided = 0;
      preparedSpec.maximumBlockSize = 0;
    }
    prepare(spec);
  }

  /** Whether the instance is prepared for blocks of this spec already. */
  bool isPreparedFor(const juce::dsp::ProcessSpec &spec) const {
    return preparedSpec.maximumBlockSize != 0 &&
           preparedSpec.sampleRate == spec.sampleRate &&
           preparedSpec.maximumBlockSize >= spec.maximumBlockSize &&
           preparedSpec.numChannels == spec.numChannels;
  }

  /**
   * The transport that the plugin sees. Position advances by itself while
   * ``playing`` is set; the host moves it on seeks and loop wraps.
   */
  class PlayHead : public juce::AudioPlayHead {
  public:
    bool getCurrentPosition(CurrentPositionInfo &info) override {
      info.resetToDefault();
      const double rate = sampleRate.load();
      const juce::int64 samples = positionSamples.load();
      const double tempo = bpm.load();
      const double seconds = rate > 0 ? (double)samples / rate : 0.0;
      const double beats = seconds * tempo / 60.0;
      info.timeInSamples = samples;
      info.timeInSeconds = seconds;
      info.isPlaying = playing.load();
      info.bpm = tempo;
      info.timeSigNumerator = 4;
      info.timeSigDenominator = 4;
      info.ppqPosition = beats;
      info.ppqPositionOfLastBarStart = std::floor(beats / 4.0) * 4.0;
      return true;
    }

    std::atomic<juce::int64> positionSamples{0};
    std::atomic<bool> playing{false};
    std::atomic<double> bpm{120.0};
    std::atomic<double> sampleRate{0.0};
  };

  PlayHead playHead;

  py::object editorPowerCallback = py::none();
  py::object editorPinCallback = py::none();
  std::unique_ptr<StandalonePluginWindow> editorWindow;

private:
  std::unique_ptr<juce::AudioPluginInstance>
  createPluginInstance(const juce::PluginDescription &foundPluginDescription,
                       double rate, int blockSize, juce::String &loadError) {
    std::unique_ptr<juce::AudioPluginInstance> instance;
    instance = pluginFormatManager.createPluginInstance(
        foundPluginDescription, rate, blockSize, loadError);
    if (!instance && loadError.contains(
                         "This plug-in cannot be instantiated synchronously")) {
      bool done = false;
      std::thread thread =
          std::thread([this, &instance, &foundPluginDescription, &rate,
                       &blockSize, &loadError, &done]() {
            instance = pluginFormatManager.createPluginInstance(
                foundPluginDescription, rate, blockSize, loadError);
            done = true;
          });

      // Pump the message thread until the scan is complete:
      while (!done) {
        juce::MessageManager::getInstance()->runDispatchLoopUntil(1);
      }

      thread.join();
    }

    return instance;
  }

  constexpr static int ExternalLoadSampleRate = 44100,
                       ExternalLoadMaximumBlockSize = 8192;
  juce::String pathToPluginFile;
  juce::AudioPluginFormatManager pluginFormatManager;
  std::unique_ptr<juce::AudioPluginInstance> pluginInstance;

  long samplesProvided = 0;
  float initializationTimeout = DEFAULT_INITIALIZATION_TIMEOUT_SECONDS;
};

inline void init_external_plugins(py::module &m) {
  py::enum_<ExternalPluginReloadType>(
      m, "ExternalPluginReloadType",
      "Indicates the behavior of an external plugin when reset() is called.")
      .value("Unknown", ExternalPluginReloadType::Unknown,
             "The behavior of the plugin is unknown. This will force a full "
             "reinstantiation of the plugin every time reset is called.")
      .value(
          "ClearsAudioOnReset", ExternalPluginReloadType::ClearsAudioOnReset,
          "This plugin clears its internal buffers correctly when reset() is "
          "called. The plugin will not be reinstantiated when reset is called.")
      .value("PersistsAudioOnReset",
             ExternalPluginReloadType::PersistsAudioOnReset,
             "This plugin does not clear its internal buffers as expected when "
             "reset() is called. This will force a full reinstantiation of the "
             "plugin every time reset is called.")
      .export_values();

  py::class_<juce::AudioProcessorParameter>(
      m, "_AudioProcessorParameter",
      "An abstract base class for parameter objects that can be added to an "
      "AudioProcessor.")
      .def("__repr__",
           [](juce::AudioProcessorParameter &parameter) {
             std::ostringstream ss;
             ss << "<pedalboard.AudioProcessorParameter";
             ss << " name=\"" << parameter.getName(512).toStdString() << "\"";
             if (!parameter.getLabel().isEmpty())
               ss << " label=\"" << parameter.getLabel().toStdString() << "\"";
             if (parameter.isBoolean())
               ss << " boolean";
             if (parameter.isDiscrete())
               ss << " discrete";
             ss << " raw_value=" << parameter.getValue();
             ss << ">";
             return ss.str();
           })
      .def_property(
          "raw_value", &juce::AudioProcessorParameter::getValue,
          &juce::AudioProcessorParameter::setValue,
          "The internal value of this parameter. Convention is that this "
          "parameter should be between 0 and 1.0. This may or may not "
          "correspond with the value shown to the user.")
      .def_property_readonly(
          "default_raw_value", &juce::AudioProcessorParameter::getDefaultValue,
          "The default internal value of this parameter. Convention is that "
          "this parameter should be between 0 and 1.0. This may or may not "
          "correspond with the value shown to the user.")
      .def(
          "get_name",
          [](juce::AudioProcessorParameter &param, int length) {
            return param.getName(length).toStdString();
          },
          py::arg("maximum_string_length"),
          "Returns the name to display for this parameter, which is made to "
          "fit within the given string length")
      .def_property_readonly(
          "name",
          [](juce::AudioProcessorParameter &param) {
            return param.getName(512).toStdString();
          },
          "Returns the name to display for this parameter at its longest.")
      .def_property_readonly(
          "label",
          [](juce::AudioProcessorParameter &param) {
            return param.getLabel().toStdString();
          },
          "Some parameters may be able to return a label string for their "
          "units. For example \"Hz\" or \"%\".")
      .def_property_readonly(
          "num_steps", &juce::AudioProcessorParameter::getNumSteps,
          "Returns the number of steps that this parameter's range should be "
          "quantised into. See also: is_discrete, is_boolean.")
      .def_property_readonly("is_discrete",
                             &juce::AudioProcessorParameter::isDiscrete,
                             "Returns whether the parameter uses discrete "
                             "values, based on the result of getNumSteps, or "
                             "allows the host to select values continuously.")
      .def_property_readonly(
          "is_boolean", &juce::AudioProcessorParameter::isBoolean,
          "Returns whether the parameter represents a boolean switch, "
          "typically with \"On\" and \"Off\" states.")
      .def(
          "get_text_for_raw_value",
          [](juce::AudioProcessorParameter &param, float value,
             int maximumStringLength) {
            return param.getText(value, maximumStringLength).toStdString();
          },
          py::arg("raw_value"), py::arg("maximum_string_length") = 512,
          "Returns a textual version of the supplied normalised parameter "
          "value.")
      .def(
          "get_raw_value_for_text",
          [](juce::AudioProcessorParameter &param, std::string &text) {
            return param.getValueForText(text);
          },
          py::arg("string_value"),
          "Returns the raw value of the supplied text. Plugins may handle "
          "errors however they see fit, but will likely not raise "
          "exceptions.")
      .def_property_readonly(
          "is_orientation_inverted",
          &juce::AudioProcessorParameter::isOrientationInverted,
          "If true, this parameter operates in the reverse direction. (Not "
          "all plugin formats will actually use this information).")
      .def_property_readonly("is_automatable",
                             &juce::AudioProcessorParameter::isAutomatable,
                             "Returns true if this parameter can be automated.")
      .def_property_readonly("is_automatable",
                             &juce::AudioProcessorParameter::isAutomatable,
                             "Returns true if this parameter can be "
                             "automated (i.e.: scheduled to "
                             "change over time, in real-time, in a DAW).")
      .def_property_readonly(
          "is_meta_parameter", &juce::AudioProcessorParameter::isMetaParameter,
          "A meta-parameter is a parameter that changes other parameters.")
      .def_property_readonly(
          "index", &juce::AudioProcessorParameter::getParameterIndex,
          "The index of this parameter in its plugin's parameter list.")
      .def_property_readonly(
          "string_value",
          [](juce::AudioProcessorParameter &param) {
            return param.getCurrentValueAsText().toStdString();
          },
          "Returns the current value of the parameter as a string.");

  py::object externalPlugin =
      py::class_<AbstractExternalPlugin, Plugin,
                 std::shared_ptr<AbstractExternalPlugin>>(
          m, "ExternalPlugin", py::dynamic_attr(),
          "A wrapper around a third-party effect plugin.\n\nDon't use this "
          "directly; use one of :class:`pedalboard.VST3Plugin` or "
          ":class:`pedalboard.AudioUnitPlugin` instead.")
          .def(py::init([]() {
            throw py::type_error(
                "ExternalPlugin is an abstract base class - don't instantiate "
                "this directly, use its subclasses instead.");
            return nullptr;
          }))
          .def(
              "process",
              [](std::shared_ptr<AbstractExternalPlugin> self,
                 py::object midiMessages, float duration, float sampleRate,
                 unsigned int numChannels, unsigned long bufferSize,
                 bool reset) -> py::array_t<float> {
                throw py::type_error("ExternalPlugin is an abstract base class "
                                     "- use its subclasses instead.");
                py::array_t<float> nothing;
                return nothing;
              },
              EXTERNAL_PLUGIN_PROCESS_DOCSTRING, py::arg("midi_messages"),
              py::arg("duration"), py::arg("sample_rate"),
              py::arg("num_channels") = 2,
              py::arg("buffer_size") = DEFAULT_BUFFER_SIZE,
              py::arg("reset") = true)
          .def(
              "process",
              [](std::shared_ptr<Plugin> self, const py::array inputArray,
                 double sampleRate, unsigned int bufferSize, bool reset) {
                return process(inputArray, sampleRate, {self}, bufferSize,
                               reset);
              },
              EXTERNAL_PLUGIN_PROCESS_DOCSTRING, py::arg("input_array"),
              py::arg("sample_rate"),
              py::arg("buffer_size") = DEFAULT_BUFFER_SIZE,
              py::arg("reset") = true)
          .def(
              "__call__",
              [](std::shared_ptr<Plugin> self, const py::array inputArray,
                 double sampleRate, unsigned int bufferSize, bool reset) {
                return process(inputArray, sampleRate, {self}, bufferSize,
                               reset);
              },
              "Run an audio or MIDI buffer through this plugin, returning "
              "audio. Alias for :py:meth:`process`.",
              py::arg("input_array"), py::arg("sample_rate"),
              py::arg("buffer_size") = DEFAULT_BUFFER_SIZE,
              py::arg("reset") = true)
          .def(
              "__call__",
              [](std::shared_ptr<AbstractExternalPlugin> self,
                 py::object midiMessages, float duration, float sampleRate,
                 unsigned int numChannels, unsigned long bufferSize,
                 bool reset) -> py::array_t<float> {
                throw py::type_error("ExternalPlugin is an abstract base class "
                                     "- use its subclasses instead.");
                py::array_t<float> nothing;
                return nothing;
              },
              "Run an audio or MIDI buffer through this plugin, returning "
              "audio. Alias for :py:meth:`process`.",
              py::arg("midi_messages"), py::arg("duration"),
              py::arg("sample_rate"), py::arg("num_channels") = 2,
              py::arg("buffer_size") = DEFAULT_BUFFER_SIZE,
              py::arg("reset") = true);

#if (JUCE_MAC || JUCE_WINDOWS || JUCE_LINUX)
  py::class_<ExternalPlugin<juce::PatchedVST3PluginFormat>,
             AbstractExternalPlugin,
             std::shared_ptr<ExternalPlugin<juce::PatchedVST3PluginFormat>>>(
      m, "VST3Plugin",
      R"(A wrapper around third-party, audio effect or instrument plugins in
`Steinberg GmbH's VST3® <https://en.wikipedia.org/wiki/Virtual_Studio_Technology>`_
format.

VST3® plugins are supported on macOS, Windows, and Linux. However, VST3® plugin
files are not cross-compatible with different operating systems; a platform-specific
build of each plugin is required to load that plugin on a given platform. (For
example: a Windows VST3 plugin bundle will not load on Linux or macOS.)

.. warning::
    Some VST3® plugins may throw errors, hang, generate incorrect output, or
    outright crash if called from background threads. If you find that a VST3®
    plugin is not working as expected, try calling it from the main thread
    instead and `open a GitHub Issue to track the incompatibility
    <https://github.com/spotify/pedalboard/issues/new>`_.


*Support for instrument plugins introduced in v0.7.4.*

*Support for running VST3® plugins on background threads introduced in v0.8.8.*
)")
      .def(
          py::init([](std::string &pathToPluginFile, py::object parameterValues,
                      std::optional<std::string> pluginName,
                      float initializationTimeout,
                      std::optional<ExternalPluginReloadType> reloadType) {
            std::shared_ptr<ExternalPlugin<juce::PatchedVST3PluginFormat>>
                plugin = std::make_shared<
                    ExternalPlugin<juce::PatchedVST3PluginFormat>>(
                    pathToPluginFile, pluginName, initializationTimeout,
                    reloadType);
            py::cast(plugin).attr("__set_initial_parameter_values__")(
                parameterValues);
            return plugin;
          }),
          py::arg("path_to_plugin_file"),
          py::arg("parameter_values") = py::none(),
          py::arg("plugin_name") = py::none(),
          py::arg("initialization_timeout") =
              DEFAULT_INITIALIZATION_TIMEOUT_SECONDS,
          // Pass the plugin's known reset behaviour (see ``_reload_type``) to
          // skip the slow probe that would otherwise determine it.
          py::arg("reload_type") = py::none())
      .def("__repr__",
           [](ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin) {
             std::ostringstream ss;
             ss << "<pedalboard.VST3Plugin";
             ss << " \"" << plugin.getName() << "\"";
             ss << " at " << &plugin;
             ss << ">";
             return ss.str();
           })
      .def("load_preset",
           &ExternalPlugin<juce::PatchedVST3PluginFormat>::loadPresetFile,
           "Load a VST3 preset file in .vstpreset format.",
           py::arg("preset_file_path"))
      .def_property(
          "preset_data",
          [](const ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin) {
            juce::MemoryBlock presetData;
            plugin.getPreset(presetData);
            return py::bytes((const char *)presetData.getData(),
                             presetData.getSize());
          },
          [](ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin,
             const py::bytes &presetData) {
            py::buffer_info info(py::buffer(presetData).request());
            plugin.setPreset(info.ptr, static_cast<size_t>(info.size));
          },
          "Get or set the current plugin state as bytes in .vstpreset "
          "format.\n\n"
          ".. warning::\n    This property can be set to change the "
          "plugin's internal state, but providing invalid data may cause the "
          "plugin to crash, taking the entire Python process down with it.")
      .def_static(
          "get_plugin_names_for_file",
          [](std::string filename) {
            return getPluginNamesForFile<juce::PatchedVST3PluginFormat>(
                filename);
          },
          "Return a list of plugin names contained within a given VST3 "
          "plugin (i.e.: a \".vst3\"). If the provided file cannot be "
          "scanned, "
          "an ImportError will be raised.")
      .def_property_readonly_static(
          "installed_plugins",
          [](py::object /* cls */) { return findInstalledVSTPluginPaths(); },
          "Return a list of paths to VST3 plugins installed in the default "
          "location on this system. This list may not be exhaustive, and "
          "plugins in this list are not guaranteed to be compatible with "
          "Pedalboard.")
      .def_property_readonly(
          "name",
          [](ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin) {
            return plugin.getName().toStdString();
          },
          "The name of this plugin.")
      .def_property(
          "raw_state",
          [](const ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin) {
            juce::MemoryBlock state;
            plugin.getState(state);
            return py::bytes((const char *)state.getData(), state.getSize());
          },
          [](ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin,
             const py::bytes &state) {
            py::buffer_info info(py::buffer(state).request());
            plugin.setState(info.ptr, static_cast<size_t>(info.size));
          },
          "A :py:class:`bytes` object representing the plugin's internal "
          "state.\n\n"
          "For the VST3 format, this is usually an XML-encoded string "
          "prefixed with an 8-byte header and suffixed with a single null "
          "byte.\n\n"
          ".. warning::\n    This property can be set to change the "
          "plugin's internal state, but providing invalid data may cause the "
          "plugin to crash, taking the entire Python process down with it.")
      .def_property_readonly(
          "descriptive_name",
          [](ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin) {
            return plugin.foundPluginDescription.descriptiveName.toStdString();
          },
          "A more descriptive name for this plugin. This may be the same as "
          "the 'name' field, but some plugins may provide an alternative "
          "name.\n\n*Introduced in v0.9.4.*")
      .def_property_readonly(
          "category",
          [](ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin) {
            return plugin.foundPluginDescription.category.toStdString();
          },
          "A category that this plugin falls into, such as \"Dynamics\", "
          "\"Reverbs\", etc.\n\n*Introduced in v0.9.4.*")
      .def_property_readonly(
          "manufacturer_name",
          [](ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin) {
            return plugin.foundPluginDescription.manufacturerName.toStdString();
          },
          "The name of the manufacturer of this plugin, as reported by the "
          "plugin itself.\n\n*Introduced in v0.9.4.*")
      .def_property_readonly(
          "version",
          [](ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin) {
            return plugin.foundPluginDescription.version.toStdString();
          },
          "The version string for this plugin, as reported by the plugin "
          "itself.\n\n*Introduced in v0.9.4.*")
      .def_property_readonly(
          "is_instrument",
          [](ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin) {
            return plugin.foundPluginDescription.isInstrument;
          },
          "True iff this plugin identifies itself as an instrument (generator, "
          "synthesizer, etc) plugin.\n\n*Introduced in v0.9.4.*")
      .def_property_readonly(
          "has_shared_container",
          [](ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin) {
            return plugin.foundPluginDescription.hasSharedContainer;
          },
          "True iff this plugin is part of a multi-plugin "
          "container.\n\n*Introduced in v0.9.4.*")
      .def_property_readonly(
          "identifier",
          [](ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin) {
            return plugin.foundPluginDescription.createIdentifierString()
                .toStdString();
          },
          "A string that can be saved and used to uniquely identify this "
          "plugin (and version) again.\n\n*Introduced in v0.9.4.*")
      .def_property_readonly(
          "reported_latency_samples",
          [](ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin) {
            return plugin.getLatencyHint();
          },
          "The number of samples of latency (delay) that this plugin reports "
          "to introduce into the audio signal due to internal buffering "
          "and processing. Pedalboard automatically compensates for this "
          "latency during processing, so this property is present for "
          "informational purposes. Note that not all plugins correctly report "
          "the latency that they introduce, so this value may be inaccurate "
          "(especially if the plugin reports 0).\n\n*Introduced in v0.9.12.*")
      .def_property_readonly(
          "_parameters",
          &ExternalPlugin<juce::PatchedVST3PluginFormat>::getParameters,
          py::return_value_policy::reference_internal)
      .def("_get_parameter",
           &ExternalPlugin<juce::PatchedVST3PluginFormat>::getParameter,
           py::return_value_policy::reference_internal)
      .def("show_editor",
           &ExternalPlugin<juce::PatchedVST3PluginFormat>::showEditor,
           SHOW_EDITOR_DOCSTRING, py::arg("close_event") = py::none(),
           py::arg("title") = py::none(), py::arg("powered") = true,
           py::arg("on_power") = py::none())
      .def(
          "process",
          [](std::shared_ptr<Plugin> self, const py::array inputArray,
             double sampleRate, unsigned int bufferSize, bool reset) {
            return process(inputArray, sampleRate, {self}, bufferSize, reset);
          },
          EXTERNAL_PLUGIN_PROCESS_DOCSTRING, py::arg("input_array"),
          py::arg("sample_rate"), py::arg("buffer_size") = DEFAULT_BUFFER_SIZE,
          py::arg("reset") = true)
      .def(
          "__call__",
          [](std::shared_ptr<Plugin> self, const py::array inputArray,
             double sampleRate, unsigned int bufferSize, bool reset) {
            return process(inputArray, sampleRate, {self}, bufferSize, reset);
          },
          "Run an audio or MIDI buffer through this plugin, returning "
          "audio. Alias for :py:meth:`process`.",
          py::arg("input_array"), py::arg("sample_rate"),
          py::arg("buffer_size") = DEFAULT_BUFFER_SIZE, py::arg("reset") = true)
      .def("process",
           &ExternalPlugin<juce::PatchedVST3PluginFormat>::renderMIDIMessages,
           EXTERNAL_PLUGIN_PROCESS_DOCSTRING, py::arg("midi_messages"),
           py::arg("duration"), py::arg("sample_rate"),
           py::arg("num_channels") = 2,
           py::arg("buffer_size") = DEFAULT_BUFFER_SIZE,
           py::arg("reset") = true)
      .def("__call__",
           &ExternalPlugin<juce::PatchedVST3PluginFormat>::renderMIDIMessages,
           "Run an audio or MIDI buffer through this plugin, returning "
           "audio. Alias for :py:meth:`process`.",
           py::arg("midi_messages"), py::arg("duration"),
           py::arg("sample_rate"), py::arg("num_channels") = 2,
           py::arg("buffer_size") = DEFAULT_BUFFER_SIZE,
           py::arg("reset") = true)
      .def_readwrite(
          "_reload_type",
          &ExternalPlugin<juce::PatchedVST3PluginFormat>::reloadType,
          "The behavior that this plugin exhibits when .reset() is called. "
          "This is an internal attribute which gets set on plugin "
          "instantiation and should only be accessed for debugging and "
          "testing.")
      .def(
          "prepare",
          [](ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin,
             double sampleRate, unsigned int bufferSize,
             unsigned int numChannels) {
            py::gil_scoped_release release;
            std::scoped_lock lock(plugin.mutex);
            plugin.prepareForPlayback(sampleRate, bufferSize, numChannels);
          },
          "Prepare the plugin for blocks of up to ``buffer_size`` samples at "
          "``sample_rate`` with ``num_channels`` channels, without passing any "
          "audio through it - what a DAW does when a plugin is inserted. Some "
          "plugins take seconds to activate; after this call the first block "
          "of that size costs nothing extra. A plugin already prepared for a "
          "different configuration is deactivated first, and its latency "
          "compensation starts over.",
          py::arg("sample_rate"), py::arg("buffer_size") = DEFAULT_BUFFER_SIZE,
          py::arg("num_channels") = 2)
      .def_property(
          "realtime",
          [](const ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin) {
            return plugin.realtime;
          },
          [](ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin,
             bool value) { plugin.setRealtime(value); },
          "Whether the plugin is told that it runs in real time, as during "
          "playback in a DAW, instead of being rendered offline. Defaults to "
          "False. Some plugins choose different algorithms, or refuse to "
          "behave like they do during playback, when rendering offline. "
          "Changing this deactivates and re-prepares the plugin before the "
          "next block, as a DAW does around an offline render, which clears "
          "its internal state: set it before processing audio.")
      .def_property(
          "playhead_position",
          [](const ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin) {
            return plugin.playHead.positionSamples.load();
          },
          [](ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin,
             juce::int64 value) { plugin.playHead.positionSamples = value; },
          "The transport position, in samples, that the plugin sees. Advances "
          "by itself with every processed block while ``playhead_playing`` is "
          "True; set it on seeks and loop wraps.")
      .def_property(
          "playhead_playing",
          [](const ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin) {
            return plugin.playHead.playing.load();
          },
          [](ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin,
             bool value) { plugin.playHead.playing = value; },
          "Whether the transport that the plugin sees is running. Defaults to "
          "False.")
      .def_property(
          "playhead_bpm",
          [](const ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin) {
            return plugin.playHead.bpm.load();
          },
          [](ExternalPlugin<juce::PatchedVST3PluginFormat> &plugin,
             double value) { plugin.playHead.bpm = value; },
          "The tempo, in beats per minute, that the plugin sees. Defaults to "
          "120.")
      .def("open_editor",
           &ExternalPlugin<juce::PatchedVST3PluginFormat>::openEditor,
           "Open the editor window and return at once. Unlike show_editor, "
           "the window is then driven by the caller: call "
           "VST3Plugin.pump_editors() regularly on this same thread, or the "
           "window will neither repaint nor react. ``title``, ``powered`` and "
           "``on_power`` are as for show_editor. With ``on_pin`` the title bar "
           "also has a pin, which opens unpinned; ``on_pin`` is called with its "
           "new state (True = pinned) on every click, and what a pin means is "
           "up to the caller. Both callbacks run from inside pump_editors(). "
           "``colors`` paints the frame: a dict of \"#rrggbb\" strings with "
           "the keys bar, line, text, text_dim, edge and accent; keys left out "
           "keep their defaults. ``resize_border`` puts a border of that many "
           "pixels around the editor of any plugin, and the window of a plugin "
           "that resizes is resized by dragging it; 0, the default, leaves "
           "only the one-pixel outline, and the plugin resizes by its own "
           "means. ``position`` puts the window's top-left corner "
           "at ``(x, y)`` before the window is shown, and ``center_on`` "
           "centers it over the rectangle ``(x, y, width, height)`` within the "
           "work area of that rectangle's display; both are in physical "
           "pixels, and ``position`` wins when both are given. Without either "
           "the window opens where JUCE puts a new window: in the top-left "
           "corner of the main display. Calling this while the window is open "
           "brings it to the front and changes nothing else.",
           py::arg("title") = py::none(), py::arg("powered") = true,
           py::arg("on_power") = py::none(), py::arg("on_pin") = py::none(),
           py::arg("colors") = py::none(), py::arg("resize_border") = 0,
           py::arg("position") = py::none(), py::arg("center_on") = py::none())
      .def("close_editor",
           &ExternalPlugin<juce::PatchedVST3PluginFormat>::closeEditor,
           "Close the window opened by open_editor(), if it is open.")
      .def_property_readonly(
          "editor_open",
          &ExternalPlugin<juce::PatchedVST3PluginFormat>::isEditorOpen,
          "Whether the window opened by open_editor() is still on screen. "
          "Becomes False once the user closes it.")
      .def_property_readonly(
          "editor_handle",
          &ExternalPlugin<juce::PatchedVST3PluginFormat>::editorHandle,
          "The native handle (HWND on Windows) of the window opened by "
          "open_editor(), or 0 when there is none.")
      .def("set_editor_powered",
           &ExternalPlugin<juce::PatchedVST3PluginFormat>::setEditorPowered,
           "Set the state of the power button in the open editor window "
           "without triggering on_power - for bypass changes made elsewhere.",
           py::arg("powered"))
      .def("set_editor_pinned",
           &ExternalPlugin<juce::PatchedVST3PluginFormat>::setEditorPinned,
           "Set the pin in the open editor window without triggering on_pin.",
           py::arg("pinned"))
      .def("set_editor_colors",
           &ExternalPlugin<juce::PatchedVST3PluginFormat>::setEditorColours,
           "Repaint the frame of the open editor window in new colors, given "
           "as for open_editor().",
           py::arg("colors"))
      .def_static(
          "pump_editors",
          &ExternalPlugin<juce::PatchedVST3PluginFormat>::pumpEditors,
          "Run the event loop of every window opened by open_editor() for "
          "the given number of milliseconds. Must be called on the thread "
          "that opened the windows.",
          py::arg("milliseconds") = 10);
#endif

#if JUCE_PLUGINHOST_AU && JUCE_MAC
  py::class_<ExternalPlugin<juce::AudioUnitPluginFormat>,
             AbstractExternalPlugin,
             std::shared_ptr<ExternalPlugin<juce::AudioUnitPluginFormat>>>(
      m, "AudioUnitPlugin", R"(
A wrapper around third-party, audio effect or instrument
plugins in `Apple's Audio Unit <https://en.wikipedia.org/wiki/Audio_Units>`_
format.

Audio Unit plugins are only supported on macOS. This class will be
unavailable on non-macOS platforms. Plugin files must be installed
in the appropriate system-wide path for them to be
loadable (usually ``/Library/Audio/Plug-Ins/Components/`` or
``~/Library/Audio/Plug-Ins/Components/``).

For a plugin wrapper that works on Windows and Linux as well,
see :class:`pedalboard.VST3Plugin`.)

.. warning::
    Some Audio Unit plugins may throw errors, hang, generate incorrect output, or
    outright crash if called from background threads. If you find that a Audio Unit
    plugin is not working as expected, try calling it from the main thread
    instead and `open a GitHub Issue to track the incompatibility
    <https://github.com/spotify/pedalboard/issues/new>`_.

*Support for instrument plugins introduced in v0.7.4.*

*Support for running Audio Unit plugins on background threads introduced in v0.8.8.*

*Support for loading AUv3 plugins (* ``.appex`` *bundles) introduced in v0.9.5.*
)")
      .def(
          py::init([](std::string &pathToPluginFile, py::object parameterValues,
                      std::optional<std::string> pluginName,
                      float initializationTimeout) {
            std::shared_ptr<ExternalPlugin<juce::AudioUnitPluginFormat>>
                plugin = std::make_shared<
                    ExternalPlugin<juce::AudioUnitPluginFormat>>(
                    pathToPluginFile, pluginName, initializationTimeout);
            py::cast(plugin).attr("__set_initial_parameter_values__")(
                parameterValues);
            return plugin;
          }),
          py::arg("path_to_plugin_file"),
          py::arg("parameter_values") = py::none(),
          py::arg("plugin_name") = py::none(),
          py::arg("initialization_timeout") =
              DEFAULT_INITIALIZATION_TIMEOUT_SECONDS)
      .def("__repr__",
           [](const ExternalPlugin<juce::AudioUnitPluginFormat> &plugin) {
             std::ostringstream ss;
             ss << "<pedalboard.AudioUnitPlugin";
             ss << " \"" << plugin.getName() << "\"";
             ss << " at " << &plugin;
             ss << ">";
             return ss.str();
           })
      .def_static(
          "get_plugin_names_for_file",
          [](std::string filename) {
            return getPluginNamesForFile<juce::AudioUnitPluginFormat>(filename);
          },
          py::arg("filename"),
          "Return a list of plugin names contained within a given Audio Unit "
          "bundle (i.e.: a ``.component`` or ``.appex`` file). If the provided "
          "file cannot be scanned, an ``ImportError`` will be raised.\n\nNote "
          "that most Audio Units have a single plugin inside, but this method "
          "can be useful to determine if multiple plugins are present in one "
          "bundle, and if so, what their names are.")
      .def_property_readonly_static(
          "installed_plugins",
          [](py::object /* cls */) {
            return AudioUnitPathFinder::findInstalledAudioUnitPaths();
          },
          "Return a list of paths to Audio Units installed in the default "
          "location on this system. This list may not be exhaustive, and "
          "plugins in this list are not guaranteed to be compatible with "
          "Pedalboard.")
      .def_property_readonly(
          "name",
          [](ExternalPlugin<juce::AudioUnitPluginFormat> &plugin) {
            return plugin.getName().toStdString();
          },
          "The name of this plugin, as reported by the plugin itself.")
      .def_property(
          "raw_state",
          [](const ExternalPlugin<juce::AudioUnitPluginFormat> &plugin) {
            juce::MemoryBlock state;
            plugin.getState(state);
            return py::bytes((const char *)state.getData(), state.getSize());
          },
          [](ExternalPlugin<juce::AudioUnitPluginFormat> &plugin,
             const py::bytes &state) {
            py::buffer_info info(py::buffer(state).request());
            plugin.setState(info.ptr, static_cast<size_t>(info.size));
          },
          "A :py:class:`bytes` object representing the plugin's internal "
          "state.\n\n"
          "For the Audio Unit format, this is usually a binary property list "
          "that can be decoded or encoded with the built-in :py:mod:`plistlib` "
          "package.\n\n"
          ".. warning::\n    This property can be set to change the "
          "plugin's internal state, but providing invalid data may cause the "
          "plugin to crash, taking the entire Python process down with it.")
      .def_property_readonly(
          "name",
          [](ExternalPlugin<juce::AudioUnitPluginFormat> &plugin) {
            return plugin.getName().toStdString();
          },
          "The name of this plugin.")
      .def_property_readonly(
          "descriptive_name",
          [](ExternalPlugin<juce::AudioUnitPluginFormat> &plugin) {
            return plugin.foundPluginDescription.descriptiveName.toStdString();
          },
          "A more descriptive name for this plugin. This may be the same as "
          "the 'name' field, but some plugins may provide an alternative "
          "name.\n\n*Introduced in v0.9.4.*")
      .def_property_readonly(
          "category",
          [](ExternalPlugin<juce::AudioUnitPluginFormat> &plugin) {
            return plugin.foundPluginDescription.category.toStdString();
          },
          "A category that this plugin falls into, such as \"Dynamics\", "
          "\"Reverbs\", etc.\n\n*Introduced in v0.9.4.*")
      .def_property_readonly(
          "manufacturer_name",
          [](ExternalPlugin<juce::AudioUnitPluginFormat> &plugin) {
            return plugin.foundPluginDescription.manufacturerName.toStdString();
          },
          "The name of the manufacturer of this plugin, as reported by the "
          "plugin itself.\n\n*Introduced in v0.9.4.*")
      .def_property_readonly(
          "version",
          [](ExternalPlugin<juce::AudioUnitPluginFormat> &plugin) {
            return plugin.foundPluginDescription.version.toStdString();
          },
          "The version string for this plugin, as reported by the plugin "
          "itself.\n\n*Introduced in v0.9.4.*")
      .def_property_readonly(
          "is_instrument",
          [](ExternalPlugin<juce::AudioUnitPluginFormat> &plugin) {
            return plugin.foundPluginDescription.isInstrument;
          },
          "True iff this plugin identifies itself as an instrument (generator, "
          "synthesizer, etc) plugin.\n\n*Introduced in v0.9.4.*")
      .def_property_readonly(
          "has_shared_container",
          [](ExternalPlugin<juce::AudioUnitPluginFormat> &plugin) {
            return plugin.foundPluginDescription.hasSharedContainer;
          },
          "True iff this plugin is part of a multi-plugin "
          "container.\n\n*Introduced in v0.9.4.*")
      .def_property_readonly(
          "identifier",
          [](ExternalPlugin<juce::AudioUnitPluginFormat> &plugin) {
            return plugin.foundPluginDescription.createIdentifierString()
                .toStdString();
          },
          "A string that can be saved and used to uniquely identify this "
          "plugin (and version) again.\n\n*Introduced in v0.9.4.*")
      .def_property_readonly(
          "reported_latency_samples",
          [](ExternalPlugin<juce::AudioUnitPluginFormat> &plugin) {
            return plugin.getLatencyHint();
          },
          "The number of samples of latency (delay) that this plugin reports "
          "to introduce into the audio signal due to internal buffering "
          "and processing. Pedalboard automatically compensates for this "
          "latency during processing, so this property is present for "
          "informational purposes. Note that not all plugins correctly report "
          "the latency that they introduce, so this value may be inaccurate "
          "(especially if the plugin reports 0).\n\n*Introduced in v0.9.12.*")
      .def_property_readonly(
          "_parameters",
          &ExternalPlugin<juce::AudioUnitPluginFormat>::getParameters,
          py::return_value_policy::reference_internal)
      .def("_get_parameter",
           &ExternalPlugin<juce::AudioUnitPluginFormat>::getParameter,
           py::return_value_policy::reference_internal)
      .def("show_editor",
           &ExternalPlugin<juce::AudioUnitPluginFormat>::showEditor,
           SHOW_EDITOR_DOCSTRING, py::arg("close_event") = py::none(),
           py::arg("title") = py::none(), py::arg("powered") = true,
           py::arg("on_power") = py::none())
      .def(
          "process",
          [](std::shared_ptr<Plugin> self, const py::array inputArray,
             double sampleRate, unsigned int bufferSize, bool reset) {
            return process(inputArray, sampleRate, {self}, bufferSize, reset);
          },
          EXTERNAL_PLUGIN_PROCESS_DOCSTRING, py::arg("input_array"),
          py::arg("sample_rate"), py::arg("buffer_size") = DEFAULT_BUFFER_SIZE,
          py::arg("reset") = true)
      .def(
          "__call__",
          [](std::shared_ptr<Plugin> self, const py::array inputArray,
             double sampleRate, unsigned int bufferSize, bool reset) {
            return process(inputArray, sampleRate, {self}, bufferSize, reset);
          },
          "Run an audio or MIDI buffer through this plugin, returning "
          "audio. Alias for :py:meth:`process`.",
          py::arg("input_array"), py::arg("sample_rate"),
          py::arg("buffer_size") = DEFAULT_BUFFER_SIZE, py::arg("reset") = true)
      .def("process",
           &ExternalPlugin<juce::AudioUnitPluginFormat>::renderMIDIMessages,
           EXTERNAL_PLUGIN_PROCESS_DOCSTRING, py::arg("midi_messages"),
           py::arg("duration"), py::arg("sample_rate"),
           py::arg("num_channels") = 2,
           py::arg("buffer_size") = DEFAULT_BUFFER_SIZE,
           py::arg("reset") = true)
      .def("__call__",
           &ExternalPlugin<juce::AudioUnitPluginFormat>::renderMIDIMessages,
           "Run an audio or MIDI buffer through this plugin, returning "
           "audio. Alias for :py:meth:`process`.",
           py::arg("midi_messages"), py::arg("duration"),
           py::arg("sample_rate"), py::arg("num_channels") = 2,
           py::arg("buffer_size") = DEFAULT_BUFFER_SIZE,
           py::arg("reset") = true)
      .def_readwrite(
          "_reload_type",
          &ExternalPlugin<juce::AudioUnitPluginFormat>::reloadType,
          "The behavior that this plugin exhibits when .reset() is called. "
          "This is an internal attribute which gets set on plugin "
          "instantiation and should only be accessed for debugging and "
          "testing.");
#endif
}

} // namespace Pedalboard
