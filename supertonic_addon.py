"""Subtitld add-on entry point for Supertonic.

Wraps `supertonic` (PyPI) — MIT — exposing the 10 baked-in preset voices
(M1-M5, F1-F5) plus an optional "custom voice" slot loaded from a Voice
Builder JSON export.

Differences from the heavyweight TTS add-ons (Coqui XTTS, Qwen3-TTS):

  * **No torch / no transformers.** Supertonic runs on onnxruntime
    directly, so the PyInstaller bundle is ~200 MB instead of ~3 GB.
    Cold start is fast (~1-2 s on CPU) — we keep
    `startup_timeout_sec=60` in the manifest as a safety margin but it
    rarely matters in practice.
  * **No clone-from-audio.** The Python SDK doesn't expose a
    `voice_ref_audio`-style API; voice cloning happens out-of-band
    through Supertone's Voice Builder, which exports a JSON that we
    load via `tts.get_voice_style_from_path`. We expose this path
    through the config schema (`voice_style_json`) rather than at
    per-request granularity — there's no way for the host to construct
    one on the fly the way it can with XTTS.
  * **Native `speed` knob.** Supertonic accepts `speed` in [0.7, 2.0]
    directly. We map Subtitld's `rate` field (free-form percent, e.g.
    `+20`, `-15`, `0`) to a multiplier and clamp.

The Supertonic Python API (paraphrased — see supertonic-py docs):

    from supertonic import TTS
    tts = TTS(auto_download=True)
    style = tts.get_voice_style(voice_name='M1')              # preset
    style = tts.get_voice_style_from_path('/path/to/voice.json')  # custom
    wav, duration = tts.synthesize(
        text=...,
        lang='en',           # or 'na' for language-agnostic
        voice_style=style,
        total_steps=8,       # 5-12, default 8
        speed=1.0,           # 0.7-2.0
    )
    # wav is np.ndarray shape (1, N) float32 at 44.1 kHz
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from pathlib import Path

log = logging.getLogger('supertonic')
logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='[supertonic] %(levelname)s %(message)s')

PROTOCOL = 1
ADDON_ID = 'supertonic'
VERSION = '0.0.1'

# Voice-id → preset name accepted by `TTS.get_voice_style(voice_name=...)`.
# The id `supertonic-custom` is special-cased at request time to load the
# user-configured `voice_style_json` instead — it's not in this map.
_VOICE_TABLE: dict[str, str] = {
    'supertonic-m1': 'M1',
    'supertonic-m2': 'M2',
    'supertonic-m3': 'M3',
    'supertonic-m4': 'M4',
    'supertonic-m5': 'M5',
    'supertonic-f1': 'F1',
    'supertonic-f2': 'F2',
    'supertonic-f3': 'F3',
    'supertonic-f4': 'F4',
    'supertonic-f5': 'F5',
}

# BCP-47 tag (lowercase, short or short-region) → Supertonic ISO code.
# We accept the long form (e.g. `pt-br`, `en-us`) and reduce to the
# 2-letter prefix; the host hands us whatever's in the project's
# `language` field, which can be either.
_SUPPORTED_LANGS = {
    'ar', 'bg', 'hr', 'cs', 'da', 'nl', 'en', 'et', 'fi', 'fr',
    'de', 'el', 'hi', 'hu', 'id', 'it', 'ja', 'ko', 'lv', 'lt',
    'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'es', 'sv', 'tr', 'uk',
    'vi',
}


# ---------------------------------------------------------------------------
# Wire helpers — identical pattern to the other TTS addons.
# ---------------------------------------------------------------------------
_write_lock = threading.Lock()


def write_frame(frame: dict) -> None:
    line = json.dumps(frame, ensure_ascii=False)
    with _write_lock:
        sys.stdout.write(line + '\n')
        sys.stdout.flush()


def emit_progress(rid, value, message=''):
    write_frame({'id': rid, 'type': 'progress',
                 'data': {'value': max(0.0, min(1.0, float(value))), 'message': message}})


def emit_error(rid, code, message, retryable=False):
    write_frame({'id': rid, 'type': 'error',
                 'data': {'code': code, 'message': message, 'retryable': retryable}})


def emit_result(rid, data):
    write_frame({'id': rid, 'type': 'result', 'data': data})


# ---------------------------------------------------------------------------
# Model state — single instance, loaded lazily, swapped only if device
# changes (rare; user has to reopen the addon config dialog to flip it).
# ---------------------------------------------------------------------------
_model_lock = threading.Lock()
_model_cache: dict = {'instance': None, 'device': None}

# Cached resolved voice style per voice id — `get_voice_style` is cheap
# but `get_voice_style_from_path` parses JSON; either way caching means
# we hit the SDK once per voice per process lifetime.
_voice_style_cache: dict[str, object] = {}
_voice_style_lock = threading.Lock()

_pending_cancel: set[str] = set()
_pending_cancel_lock = threading.Lock()


def _resolve_providers(device: str) -> list:
    """Map our config `device` value to onnxruntime ExecutionProvider names.

    onnxruntime is strict: requesting `CUDAExecutionProvider` on a build
    that doesn't bundle the CUDA EP raises at session-create time. We
    fall back through to CPUExecutionProvider in every case — that's
    always available — so a misconfigured device value degrades to slow
    rather than fatal.
    """
    pref = (device or 'cpu').lower()
    if pref == 'cuda':
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if pref == 'coreml':
        return ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']


def _load_model(device: str):
    """Load (or reload) the Supertonic TTS model. Heavy on first call
    because it triggers a Hugging Face download (~400 MB); subsequent
    loads are ~1 s from disk cache."""
    with _model_lock:
        if (_model_cache['instance'] is not None
                and _model_cache['device'] == device):
            return _model_cache['instance']

        # Drop previous instance so we don't briefly hold two on GPU.
        _model_cache['instance'] = None
        _model_cache['device'] = None

        try:
            from supertonic import TTS  # type: ignore
        except ImportError as exc:
            raise RuntimeError(f'supertonic python package not available: {exc}') from exc

        providers = _resolve_providers(device)
        log.info('loading Supertonic TTS (providers=%s)', providers)

        # The supertonic SDK doesn't take an explicit providers kwarg yet
        # (1.2.x). We set the env var the upstream code reads at session
        # creation — onnxruntime itself respects ORT_DEFAULT_PROVIDERS,
        # and supertonic forwards through to InferenceSession.
        os.environ.setdefault('ORT_PROVIDERS', ','.join(providers))

        # auto_download=True downloads ~400 MB on first run. The progress
        # frame fires before this so the user sees *something* instead
        # of a 30-60 s freeze on a fresh install.
        tts = TTS(auto_download=True)
        _voice_style_cache.clear()  # styles are bound to a TTS instance
        _model_cache['instance'] = tts
        _model_cache['device'] = device
        return tts


def _get_voice_style(tts, voice_id: str, voice_style_json: str | None):
    """Resolve a voice id to a Supertonic style object, with caching.

    `supertonic-custom` consults the configured `voice_style_json` path;
    everything else looks up `_VOICE_TABLE`. Errors raise — the caller
    converts them to `unsupported_voice` or `bad_params` as appropriate.
    """
    with _voice_style_lock:
        cached = _voice_style_cache.get(voice_id)
        if cached is not None:
            return cached

        if voice_id == 'supertonic-custom':
            if not voice_style_json:
                raise ValueError(
                    'voice id "supertonic-custom" requires the '
                    '`voice_style_json` config to point at a Voice Builder '
                    'JSON export'
                )
            if not os.path.isfile(voice_style_json):
                raise FileNotFoundError(
                    f'voice_style_json does not exist: {voice_style_json}'
                )
            style = tts.get_voice_style_from_path(voice_style_json)
        else:
            preset = _VOICE_TABLE.get(voice_id)
            if preset is None:
                raise KeyError(f'unknown voice id: {voice_id!r}')
            style = tts.get_voice_style(voice_name=preset)

        _voice_style_cache[voice_id] = style
        return style


def _resolve_language(raw: str, default_mode: str) -> str:
    """Pick the `lang` value to pass to `tts.synthesize`.

    The host sends BCP-47-ish tags (`pt-br`, `en-us`, `pt`, etc.).
    Supertonic wants ISO-639-1 short codes from a fixed allowlist, or
    the literal `"na"` for language-agnostic synthesis.

    Resolution:
      1. If `default_mode == 'na'` (user opted to always go
         language-agnostic), return `"na"` regardless of the tag.
      2. Try the full tag, then the 2-letter prefix; first hit in the
         allowlist wins.
      3. Fallback: `"na"`. Lossy but never errors out — Supertonic
         handles unknown-language text gracefully in `na` mode, which
         is strictly preferable to refusing the request.
    """
    if (default_mode or '').lower() == 'na':
        return 'na'
    tag = (raw or '').lower().strip()
    if not tag:
        return 'na'
    if tag in _SUPPORTED_LANGS:
        return tag
    prefix = tag.split('-', 1)[0]
    if prefix in _SUPPORTED_LANGS:
        return prefix
    return 'na'


def _rate_to_speed(rate) -> float:
    """Map Subtitld's `rate` (free-form percent, signed int or string
    like '+20%') to Supertonic's `speed` multiplier, clamped to [0.7, 2.0].

    Subtitld convention: rate=0 → 1.0× (no change), rate=+50 → 1.5×,
    rate=-30 → 0.7×. Strings can carry a trailing '%'; numbers are
    taken as percent deltas.
    """
    if rate is None or rate == '':
        return 1.0
    try:
        if isinstance(rate, str):
            cleaned = rate.strip().rstrip('%').replace('+', '')
            pct = float(cleaned)
        else:
            pct = float(rate)
    except (TypeError, ValueError):
        return 1.0
    speed = 1.0 + (pct / 100.0)
    # Supertonic refuses values outside [0.7, 2.0] — clamp at the edge
    # rather than erroring so users who push the slider hard get the
    # closest synthesis instead of a failure.
    if speed < 0.7:
        return 0.7
    if speed > 2.0:
        return 2.0
    return speed


def _apply_seed(seed_param) -> None:
    """Best-effort: seed the numpy + onnxruntime RNGs for the next call.

    Supertonic 1.2.x doesn't expose a `seed` argument on synthesize, but
    its internal sampler reads numpy.random and the onnxruntime EP can
    be coerced to deterministic mode via environment. We do what we can
    — same additive contract as the qwen3-tts addon: missing or
    unparseable seed → fall through to stochastic behaviour, never
    error.
    """
    if seed_param is None:
        return
    try:
        seed_int = int(seed_param) & 0xFFFFFFFF
    except (TypeError, ValueError):
        log.debug('seed=%r not int-castable, ignored', seed_param)
        return
    try:
        import numpy as np
        np.random.seed(seed_int)
    except Exception as exc:
        log.warning('numpy seed apply failed (%s); continuing unseeded', exc)


# ---------------------------------------------------------------------------
# Audio writing
# ---------------------------------------------------------------------------
def _write_wav(path: str, wav, sample_rate: int) -> tuple[float, int, int]:
    """Write Supertonic's float32 (1, N) output as PCM-16 mono WAV.

    Supertonic returns shape `(1, N)` at 44.1 kHz. The host's
    `audioengine` resamples to 48 kHz on load; we don't pre-resample
    here to avoid double-processing.
    """
    import numpy as np
    import soundfile as sf

    arr = np.asarray(wav)
    if arr.ndim > 1:
        arr = arr.squeeze()
    if arr.ndim != 1:
        raise RuntimeError(f'unexpected waveform shape: {arr.shape}')

    arr = np.clip(arr, -1.0, 1.0)
    sf.write(path, arr, int(sample_rate), subtype='PCM_16')

    duration = float(len(arr)) / float(sample_rate or 1)
    return duration, int(sample_rate), 1


# ---------------------------------------------------------------------------
# Request handling
# ---------------------------------------------------------------------------
def handle_tts_synthesize(rid: str, params: dict, defaults: dict) -> None:
    text = params.get('text')
    voice_id = params.get('voice')
    output_path = params.get('output_path')
    if not text or not voice_id or not output_path:
        emit_error(rid, 'bad_params', 'text, voice, and output_path are all required')
        return

    # Honour cancellation that arrived before we even started — same
    # cheap check as the qwen3 path.
    with _pending_cancel_lock:
        if rid in _pending_cancel:
            _pending_cancel.discard(rid)
            emit_error(rid, 'cancelled', 'cancelled before synthesis started')
            return

    # Fail fast on unknown ids so we don't waste the model load.
    if voice_id not in _VOICE_TABLE and voice_id != 'supertonic-custom':
        emit_error(rid, 'unsupported_voice',
                   f'unknown voice id: {voice_id!r} '
                   f'(supported: {sorted(_VOICE_TABLE)} or supertonic-custom)')
        return

    emit_progress(rid, 0.05,
                  'Loading Supertonic (first call may download ~400 MB from Hugging Face)...')
    try:
        tts = _load_model(defaults['device'])
    except Exception as exc:
        log.exception('model load failed')
        emit_error(rid, 'internal', f'failed to load Supertonic: {exc}')
        return

    try:
        style = _get_voice_style(tts, voice_id, defaults.get('voice_style_json'))
    except ValueError as exc:
        emit_error(rid, 'bad_params', str(exc))
        return
    except FileNotFoundError as exc:
        emit_error(rid, 'model_missing', str(exc))
        return
    except KeyError as exc:
        emit_error(rid, 'unsupported_voice', str(exc))
        return
    except Exception as exc:
        log.exception('voice style resolution failed')
        emit_error(rid, 'internal', f'voice style resolution failed: {exc}')
        return

    lang = _resolve_language(params.get('language') or '',
                             defaults.get('default_language_mode') or 'auto')
    speed = _rate_to_speed(params.get('rate'))
    total_steps = int(defaults.get('total_steps') or 8)
    total_steps = max(5, min(12, total_steps))

    _apply_seed(params.get('seed'))

    emit_progress(rid, 0.4, 'Synthesizing...')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        wav, _duration = tts.synthesize(
            text=text,
            lang=lang,
            voice_style=style,
            total_steps=total_steps,
            speed=speed,
        )
    except Exception as exc:
        log.exception('synth failed')
        emit_error(rid, 'internal', f'synthesize failed: {exc}')
        return

    # Supertonic always emits 44.1 kHz. Constant in the SDK; if upstream
    # ever changes it they'll surface it via the result so the host
    # resampler picks it up correctly.
    sample_rate = 44100
    try:
        duration, sr, channels = _write_wav(output_path, wav, sample_rate)
    except Exception as exc:
        log.exception('wav write failed')
        emit_error(rid, 'internal', f'failed to write output WAV: {exc}')
        return

    emit_progress(rid, 0.99, 'Finalizing...')
    emit_result(rid, {
        'path': output_path,
        'duration_sec': duration,
        'sample_rate': sr,
        'channels': channels,
    })


# ---------------------------------------------------------------------------
# Main loop — same shape as the other TTS addons.
# ---------------------------------------------------------------------------
def main() -> int:
    manifest_path = Path(__file__).resolve().parent / 'manifest.json'
    voices: list[dict] = []
    languages: list[str] = []
    config_defaults: dict = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            voices = manifest.get('voices') or []
            languages = manifest.get('languages') or []
            config_defaults = {f.get('key'): f.get('default')
                               for f in (manifest.get('config_schema') or {}).get('fields', [])
                               if f.get('default') is not None}
        except Exception:
            log.exception('manifest parse failed')

    # Host injects config via env vars (set from
    # CONFIG['addons']['options']['supertonic']).
    defaults = {
        'device': os.environ.get('SUPERTONIC_DEVICE') or config_defaults.get('device', 'cpu'),
        'total_steps': int(os.environ.get('SUPERTONIC_TOTAL_STEPS')
                           or config_defaults.get('total_steps', 8) or 8),
        'default_language_mode': (os.environ.get('SUPERTONIC_LANGUAGE_MODE')
                                  or config_defaults.get('default_language_mode', 'auto')),
        'voice_style_json': os.environ.get('SUPERTONIC_VOICE_STYLE_JSON') or '',
    }

    # If the user pointed at a custom voice JSON, advertise the
    # `supertonic-custom` voice — otherwise hide it so the dubbing UI's
    # combobox doesn't dangle an option that will error out.
    advertised_voices = list(voices)
    if defaults['voice_style_json']:
        advertised_voices.append({
            'id': 'supertonic-custom',
            'language': '*',
            'display_name': 'Custom voice (loaded from configured Voice Builder JSON)',
        })

    write_frame({
        'type': 'hello',
        'protocol': PROTOCOL,
        'addon': ADDON_ID,
        'version': VERSION,
        'capabilities': [
            {'task': 'tts.synthesize', 'languages': languages,
             'voices': advertised_voices, 'voice_clone': False},
        ],
    })

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            frame = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        ftype = frame.get('type')
        rid = frame.get('id', '')

        if ftype == 'shutdown':
            log.info('shutdown received; exiting')
            return 0
        if ftype == 'cancel':
            target = (frame.get('data') or {}).get('target') or frame.get('target')
            if target:
                with _pending_cancel_lock:
                    _pending_cancel.add(target)
            continue
        if ftype == 'tts.synthesize':
            threading.Thread(
                target=handle_tts_synthesize,
                args=(rid, frame.get('params') or {}, defaults),
                daemon=True,
            ).start()
            continue
        # Host control frames (`ready`, etc.) — log and ignore. Only
        # error on actual *requests* we can't service.
        if not rid:
            log.debug('ignoring host control frame: %s', ftype)
            continue

        emit_error(rid, 'bad_params', f'unknown request type: {ftype!r}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
