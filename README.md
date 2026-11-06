# Injini Floor Room

**A plant room where the machines have a voice.** A fitter holds a phone to a pump and drops the recording into the plant's Discord channel. Injini, an acoustic agent, scores it against that machine's healthy baseline and posts what it heard. When a machine sounds wrong, Injini opens a case room, the Foreman writes the work order, and the stop decision goes to a person on an approval card. A recording after the repair closes the case.

Built for the [Flip the Switch hackathon](https://www.wemakedevs.org/hackathons/switch), Track 1: Best Business Workflow.

## Why

Mills, mines, cold rooms and borehole pumps across Southern Africa run without vibration sensors. Every fitter carries a phone. Injini turns those phones into condition monitoring, and Switch turns the result into teamwork: evidence, a case, a work order and a human decision, in the chat the team already uses.

## The workflow

```
Discord #plant-floor ──voice note──▶ Switch room ──▶ injini (Claude Code + acoustic model)
                                                     │  python injini.py score  →  4.3σ, 330–851 Hz
                                                     ├─ posts the spectrogram
                                                     ├─ create_room  →  #case-pump-00-001 (linked to the plant floor)
                                                     ├─ case-file document
                                                     ├─ @foreman  →  work-order document, handoff to @artisan
                                                     └─ python injini.py stop  →  Switch approval card  →  supervisor
after the repair ──voice note──▶ case room ──▶ injini re-scores  →  −3.6σ  →  case closed
```

| Switch feature | What it does here |
|---|---|
| Discord bridge | Voice notes arrive as files, spectrograms go back as images |
| Claude Code agents | `injini` hosts the acoustic model, `foreman` plans the repair |
| Rooms created at runtime | One case room per fault, linked to the plant floor |
| Room documents | `case-file` and `work-order` live in the case room |
| Roles | `@artisan`, an exclusive on-call role for the repair |
| Approval cards | The stop command always asks; only people allowed to address Injini can answer |

## The ear

`injini_floor/` reuses the [Injini](https://github.com/rapha18th/injini) recipe: a frozen AudioSet-pretrained embedder (EfficientAT `mn10_as`, ONNX), whitened distance in embedding space.

- **Enrol** a machine from 3 healthy 10-second clips. Leave-one-clip-out Ledoit-Wolf Gaussians; new clips are scored by the same fold models.
- **Score** in sigma: the clip's median window distance against a bootstrapped spread of held-out healthy clip medians. Alarm at 3σ.
- **Explain**: each of 8 mel band groups is replaced with real healthy frames; the band that removes the most sigma is the one that moved. A spectral diff finds the peak inside it.

### Honest numbers

DCASE 2020 Task 2 dev set (MIMII recordings), 3 enrolment clips per machine, 20 healthy and 20 faulty test clips each:

| Machine | AUC | Healthy clips flagged at 3σ | Faults flagged at 3σ |
|---|---|---|---|
| pump id_00 | 0.83 | 0 / 20 | 7 / 20 |
| pump id_02 | 0.83 | 0 / 20 | 0 / 20 |
| pump id_04 | 0.80 | 3 / 20 | 9 / 20 |
| pump id_06 | 0.59 | 1 / 20 | 6 / 20 |
| fans id_00–06 | 0.36–0.59 | | |

Three clips is enough to rank pump faults well. Fan faults in this set sit at chance with so little data, so the demo machine is a pump. The demo fault clip is the strongest of pump id_00's twenty (+4.3σ after Opus encoding).

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q tests                     # real pump clips: fault out, healthy in
cd agents/injini
python injini.py enrol pump-00 ../../samples/pump-00-healthy-{1,2,3}.ogg
python injini.py score pump-00 ../../samples/pump-00-fault.ogg
```

In Switch:

1. Install [Switch Console](https://github.com/sandbox-quantum/switch/releases) and run a server on this computer.
2. Connect Discord on the server's Home page, then post in `#plant-floor`.
3. Add two Claude Code agents: `injini` in `agents/injini`, `foreman` in `agents/foreman`. Leave auto-approve off for `injini`.
4. `/invite-agent injini` and `/invite-agent foreman` in `#plant-floor`.
5. Drop `samples/pump-00-fault.ogg` into the channel with `@injini pump-00`.

`agents/injini/.claude/settings.json` lets Injini score and enrol freely and puts `stop` behind an approval, which Switch renders as the card.

`injini_floor/agent.py` and `switch_client.py` are a raw agent-protocol version of the same workflow (SSE in, HTTP out, cards via `/agent-sessions/{id}/approvals`) for servers that issue registration keys.

## Data

Sample clips: MIMII / DCASE 2020 Challenge Task 2 development set (Purohit et al., 2019; Koizumi et al., 2020), via the Hugging Face mirror `LakoreAI/stgram-mfn-dcase2020-dev`, Opus-encoded. Used for non-commercial demonstration under the dataset's licence.

## License

MIT
