# supertonic add-on for Subtitld

Lightning-fast on-device TTS backed by
[supertonic](https://github.com/supertone-inc/supertonic) (MIT wrapper,
OpenRAIL-M weights). The lightest of Subtitld's neural TTS add-ons —
~200 MB bundle, ~400 MB model download, no GPU required — covering 31
languages with 10 preset voices.

## When to install

- you want offline TTS but don't need the heavyweight (~3 GB) downloads
  Coqui XTTS or Qwen3-TTS require,
- your project's source language is in the [31-language list](#languages)
  but isn't covered well by the lighter Piper voices,
- you'd rather pick a preset voice than wrangle a clone reference clip.

Switch to **Coqui XTTS** or **Qwen3-TTS** instead when:

- you need to clone a specific voice from a short reference clip (Voice
  Builder's offline cloning flow is the only way to get a custom voice
  into this add-on),
- you need pitch / emotion control beyond Supertonic's speed knob and
  inline expression tags.

## Languages

Arabic (`ar`), Bulgarian (`bg`), Croatian (`hr`), Czech (`cs`), Danish
(`da`), Dutch (`nl`), English (`en`), Estonian (`et`), Finnish (`fi`),
French (`fr`), German (`de`), Greek (`el`), Hindi (`hi`), Hungarian
(`hu`), Indonesian (`id`), Italian (`it`), Japanese (`ja`), Korean
(`ko`), Latvian (`lv`), Lithuanian (`lt`), Polish (`pl`), Portuguese
(`pt`), Romanian (`ro`), Russian (`ru`), Slovak (`sk`), Slovenian
(`sl`), Spanish (`es`), Swedish (`sv`), Turkish (`tr`), Ukrainian
(`uk`), Vietnamese (`vi`).

Set **Language hint** to *Language-agnostic* in the addon config when
the project source language is uncertain — Supertonic processes the
text language-agnostically and is generally robust to mixed-language
input.

## Voices

10 preset timbres baked into the model:

| Voice id          | Type   |
| ----------------- | ------ |
| `supertonic-m1`-`m5` | Male preset 1-5 |
| `supertonic-f1`-`f5` | Female preset 1-5 |

The presets are timbre-only — voice character stays consistent across
languages, but pronunciation switches per the `lang` parameter the host
sends.

### Custom voices via Voice Builder

Supertone offers an [out-of-band Voice Builder](https://supertonic.supertone.ai/voice_builder)
that turns a short voice sample into a JSON export. Drop the file path
into the addon's **Custom voice JSON** config field and the addon
exposes one additional voice id (`supertonic-custom`) that loads it. We
don't accept clone reference audio at request time — there's no
runtime cloning API in the supertonic Python SDK.

## Models

| Model | Size | Notes |
| --- | --- | --- |
| `Supertone/supertonic-3` (default) | ~400 MB | 99M params, 44.1 kHz output, 31 languages. Auto-downloaded on first use. |

Cache lives under `~/.cache/supertonic3/` by default — override with
`SUPERTONIC_CACHE_DIR=...`.

## Building

```bash
pip install pyinstaller
pip install supertonic
pyinstaller supertonic-addon.spec --distpath dist/
cd dist/supertonic-addon
zip -r ../supertonic-0.0.1-linux-x86_64.zip . ../../manifest.json ../../LICENSE ../../README.md
```

## Output contract

Single result frame per request:

```json
{"path": "<path>", "duration_sec": <float>, "sample_rate": 44100, "channels": 1}
```

Supertonic emits 44.1 kHz mono float32; the addon writes it out as
PCM-16 mono WAV. Subtitld's host resamples to 48 kHz on `load_audio`.

## License

Wrapper code: MIT. `supertonic` PyPI package: MIT. Supertonic-3 model
weights: OpenRAIL-M — review the [model card](https://huggingface.co/Supertone/supertonic-3)
before commercial use.
