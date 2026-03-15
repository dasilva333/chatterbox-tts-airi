# Chatterbox Usage

This document covers the day-to-day workflow for running Chatterbox with AIRI and managing the preset/profile library.

## Concepts

### Base voices
Base voices are the raw clips stored in `voices/`. These are the actual voice-cloning sources such as `ivy`, `lain2`, or `zenbara`.

### Presets
Presets are reusable "virtual voices". A preset bundles:
- a base voice file
- a TTS mode (`full` or `turbo`)
- an exaggeration level
- a linked profile
- UI-only discovery hints like `ui_expressions` and `ui_mannerisms`

Use presets when you want AIRI to treat a character voice as a named reusable package instead of configuring the raw voice each time.

### Profiles
Profiles transform raw text before synthesis. They are where the character flavor lives.

Typical profile behaviors:
- replacing `~` with fillers like `nya`, `woof`, or `desu`
- collapsing hmph-like words into a chosen utterance
- mapping emoticons like `0_0` into spoken sounds like `[meow]`

## AIRI Studio Workflow

When AIRI is pointed at this server through the Chatterbox speech provider, the provider page becomes the management surface.

Recommended workflow:
1. Create or edit a profile first.
2. Save the profile.
3. Create or edit a preset that points at that profile.
4. Save the preset.
5. Use the built-in speech playground in AIRI to test the result immediately.

The AIRI studio talks to these endpoints:
- `GET /chatterbox/capabilities`
- `GET /chatterbox/presets`
- `POST /chatterbox/presets`
- `PUT /chatterbox/presets/{id}`
- `DELETE /chatterbox/presets/{id}`
- `GET /chatterbox/profiles`
- `POST /chatterbox/profiles`
- `PUT /chatterbox/profiles/{id}`
- `DELETE /chatterbox/profiles/{id}`

## Manual Editing

You can still edit `presets.json` and `profiles.json` directly if you want. The server hot-reloads both files automatically on the next request.

Use manual editing when:
- bulk-editing many entries
- recovering from a bad studio draft
- copying configs between machines

## Runtime Notes

- Deleting or renaming a profile that is still used by a preset is blocked by the server.
- Native voices and presets are merged into the `/v1/voices` response so OpenAI-compatible clients can discover both.
- The AIRI management studio depends on the CRUD release:
  - AIRI fork: `7c5ec4b1`
  - Chatterbox fork: `dd6d484`

## Minimal Start Command

```bash
run_server.bat --mannerisms=catgirl
```

## Turbo Start Command

```bash
run_server.bat --mannerisms=catgirl --turbo
```
