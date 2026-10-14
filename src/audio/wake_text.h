#ifndef BIBIT_AUDIO_WAKE_TEXT_H_
#define BIBIT_AUDIO_WAKE_TEXT_H_

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

struct NormalizedText {
  std::string text;
  std::vector<size_t> source_ends;
};

// Drops punctuation and whitespace, lowercases ASCII, and records where every
// kept byte ended in the source. CJK bytes are copied as they are.
inline NormalizedText normalizeWakeText(const std::string &source) {
  NormalizedText out;
  for (size_t i = 0; i < source.size();) {
    const unsigned char c = static_cast<unsigned char>(source[i]);
    if (c < 0x80 && (std::isspace(c) || c == ',' || c == '.' || c == '!' ||
                     c == '?' || c == ':' || c == ';' || c == '_' || c == '-')) {
      ++i;
      continue;
    }
    bool skipped = false;
    for (const std::string &separator : {"，", "。", "！", "？", "、", "：", "；"}) {
      if (source.compare(i, separator.size(), separator) == 0) {
        i += separator.size();
        skipped = true;
        break;
      }
    }
    if (skipped) continue;
    if (c < 0x80) {
      char lowered = static_cast<char>(c);
      if (lowered >= 'A' && lowered <= 'Z') lowered = static_cast<char>(lowered - 'A' + 'a');
      out.text.push_back(lowered);
      out.source_ends.push_back(i + 1);
      ++i;
    } else {
      size_t width = 1;
      if ((c & 0xE0) == 0xC0) width = 2;
      else if ((c & 0xF0) == 0xE0) width = 3;
      else if ((c & 0xF8) == 0xF0) width = 4;
      width = std::min(width, source.size() - i);
      for (size_t j = 0; j < width; ++j) {
        out.text.push_back(source[i + j]);
        out.source_ends.push_back(i + j + 1);
      }
      i += width;
    }
  }
  return out;
}

// Spellings of the wake word's second syllable that count as a match. The
// shape depends on the loaded ASR: SenseVoice emits latin words, while
// Qwen3-ASR transliterates short English phrases into CJK ("hello，B比特。").
// Its language hint does not help - `en`, `zh,en` and `auto` all produced the
// same CJK phrasing, so the table has to carry both scripts.
// ponytail: one flat list; split it per model only if an ASR swap really needs
// a different set rather than one more entry.
inline constexpr std::string_view kWakeNameVariants[] = {
    "bibit", "bibi", "bbit", "bebit", "hellobi", "比特",
};

// Both the greeting and one of the spellings above have to be present. A
// truncated clip often transcribes to a bare "hello", which must not pass.
inline bool isFallbackWakeText(const std::string &text) {
  const NormalizedText normalized = normalizeWakeText(text);
  const std::string &value = normalized.text;
  if (value.find("hello") == std::string::npos) return false;
  for (const std::string_view variant : kWakeNameVariants) {
    if (value.find(variant) != std::string::npos) return true;
  }
  return false;
}

#endif  // BIBIT_AUDIO_WAKE_TEXT_H_
