# Elkit live demo: script (about 6 minutes)

**Before you start:** Elkit is running (`systemctl --user start fda.service`), volume 70 %, Wi-Fi is up, and `docs/project-budget.csv` is on the Pi.
Speak normally, about an arm's length from the mic. Elkit's lines are approximate: the LLM words them a little differently each time.

| # | You say | Elkit should… | If it doesn't |
|---|---|---|---|
| **1. Hello** | "Hi Elkit, how are you?" | Answer in a sentence and ask back. The wave under its name shows it heard you. | Say it again, a bit closer to the mic. |
| **2. Joke** | "Tell me a joke." | Ask the setup question, then **wait**. | — |
| | "Why?" | Say the punchline. If you stay quiet, it says "This is the part where you say, why?" | — |
| **3. Virtual Home** | "Turn on the lights." | Switch to the Virtual Home tab: "Lights on." The lamp lights the room. | — |
| | "Make them warm white." | The room turns warm. | "Set the lights to warm white." |
| | "Dim the lights to 40 percent." | "Lights at 40 percent." | — |
| | "Turn the lights blue… wait, no, red." | "Okay, not blue: red instead." Blue never shows. *(change of mind)* | — |
| | "It's a bit warm. Lower the thermostat to 22 degrees." | "Thermostat set to 22 degrees." The dial moves. | "Set the thermostat to 22." |
| **4. Virtual Office** | "Go to the office." | Switch to the Virtual Office tab right away. | "Open the Virtual Office tab." |
| | "What does my project budget look like?" | Read `project-budget.csv` (offline): about **₱85,000 total, ₱34,150 spent, ₱50,850 left**, and Davao travel (₱18,000) still to book. | "Check my budget file." |
| | "Draft a meeting agenda for the Smart Room Pilot meeting at 3 PM today. Include the budget numbers." | Build an agenda document in the Office tab, live, within seconds. | If the numbers are missing: "Add the budget table to the agenda." |
| | "Make the title bigger." | Edit the doc live. | — |
| **5. Flight** | "While you're on it, check the cheapest flight from Manila to Davao." | Search the web, then say the **price, airline and link**. The links appear on screen. | "Search for cheap flights Manila to Davao." |
| | *(optional)* Talk over it: "Okay, stop." | Stop mid-sentence (barge-in). | — |
| **6. Reminder** | "Remind me at 5 PM to book the Davao flight." | "Over to the Virtual Home tab." A reminder note is pinned for 5 PM. | "Add a reminder: book the Davao flight, 5 PM." |
| **7. YouTube** | "Please open a YouTube video for me." | Ask what you'd like to watch. It never picks a video for you. | — |
| | "The latest Worlds 2026 video." | Switch to the Virtual Studio tab ("Over to the Virtual Studio tab.") and play it. | "Play the Worlds 2026 highlights on YouTube." |
| | "Elkit, pause." · "Elkit, resume." · "Elkit, louder." | Control the video. During a video, say **"Elkit"** first. | — |
| | "Elkit, close the video." | Close the video's tab. | — |
| **Close** *(optional)* | "Call my wife." | The phone on the side table rings; Mommy doesn't pick up. Elkit offers a text. | — |
| | "Yes, text her: I'll be home late." | A text is "sent" (simulated, no SIM). | — |

**Say while it works (talking points):**
- Everything runs on the Raspberry Pi 5: Silero VAD, Pipecat, and the MCP servers for ASR, TTS and tools (39 tools). The voice is Piper, offline. The LLM and ASR use Groq.
- Full duplex: you can talk over it, and "mm-hm" or side talk doesn't stop it.
- Measured on the Pi: end of speech → first audio 2.94 s mean; TTS 0.48 s; 0 % WER on the 30 test phrases.

**If something goes wrong:** say "Elkit, stop", then repeat the line more simply. If a video is playing, start with "Elkit, …". If Elkit stays silent for over 10 s, check the Wi-Fi (Groq needs internet).
