# AI Lab HSKA: Reinforcement Learning

Praktische Herangehensweise an die Lösung unterschiedlich komplexer Kontroll-Probleme mit Hilfe von modernen Reinforcement Learning (RL) Algorithmen. Von tabellarischen Methoden wie Q-Learning bis hin zur Funktionsapproximation durch Neuronale Netze soll versucht werden, Agenten in verschiedenen OpenAI Gym Umgebungen zu trainieren. Das Hauptziel am Ende ist es ein Atari Game mit Deep RL zu bewältigen.

## Vorbereitung

### System ohne nvidia GPU

Docker Image bauen:
```bash
docker build -f cpu.Dockerfile -t ai-lab-rl .
```

Jupyter Notebook inkl. TensorBoard starten:
```bash
docker-compose up ai-lab-rl-cpu
```

### System mit nvidia GPU

Docker Image bauen:
```bash
docker build -f gpu.Dockerfile -t ai-lab-rl .
```

Jupyter Notebook inkl. TensorBoard starten:
```bash
docker-compose up ai-lab-rl-gpu
```

### System ohne h264 und co
Miniconda herunterladen und installieren:
```bash
bash get_miniconda.sh
```

Conda environment erstellen:
```bash
conda env create -f conda_env_gpu_vp9.yml
```

Conda environment aktivieren:
```bash
source activate ai-lab-hska-rl
```

Agent.py direkt von der Kommandozeile starten:
```bash
PYTHONPATH=".:{$PYTHONPATH}" python notebooks/session_4/agent.py
```

Video der besten Episode anschauen: 
Es wird die Datei `play_best_episode.html` erstellt, einfach im Browser öffnen.
Sowohl der `video` Ordner als auch die `html` Datei werden relativ zum Ausführungsort des gestarteten Pythonprozesses angelegt.

### Jupyter Lab

Im Browser `http://localhost:8888` aufrufen.
**Speichern klappt nur bei Tusted Notebooks!**

### TensorBoard

Im Browser `http://localhost:6006` aufrufen.

## Termine

| Datum | Inhalt |
|-|-|
| 13.12. | GridWorld mit Q-Learning |
| 20.12. | CartPole Gym mit Q-Learning + DQN |
| 10.01. | Atari Gym mit DQN (pixel-basierter State-Space) |
| 17.01. | Atari Gym mit PPO |
| 24.01. | Atari Gym mit PPO (Fortsetzung) |

## Hinweise

### Alle Ausgabezellen löschen

```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace Notebook.ipynb
```

## Weiterführende Lektüre

- "Standard"-Lektüre für den Einstieg in RL: [Reinforcement Learning: An Introduction (Richard S. Sutton and Andrew G. Barto)](http://incompleteideas.net/book/RLbook2018.pdf)
- Ausführlich und gut erklärter Einstieg in RL (Video-Lektionen) von David Silver (Google DeepMind): [UCL Course on RL (David Silver)](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html)
- [Algorithms in Reinforcement Learning (Csaba Szepesvári)](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf)