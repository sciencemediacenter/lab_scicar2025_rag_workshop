<div id="header" align="center">
  <img src="https://media.sciencemediacenter.de/static/img/logos/smc/smc-logo-typo-bw-big.png" width="300"/>

  <div id="badges" style="padding-top: 20px">
    <a href="https://www.sciencemediacenter.de">
      <img src="https://img.shields.io/badge/Website-orange?style=plastic" alt="Website Science Media Center"/>
    </a>
    <a href="https://lab.sciencemediacenter.de">
      <img src="https://img.shields.io/badge/Website (SMC Lab)-grey?style=plastic" alt="Website Science Media Center Lab"/>
    </a>
    <a href="https://twitter.com/smc_germany_lab">
      <img src="https://img.shields.io/badge/Twitter-blue?style=plastic&logo=twitter&logoColor=white" alt="Twitter SMC Lab"/>
    </a>
  </div>
</div>

# RAG-Demo 

Dieses Repository enthält die Materialien und Code-Beispiele zum Vortrag  **Von Daten zur RAG: Ein Praxiseinblick in die Entwicklung RAG-basierter Chatbots** für die SciCAR 2025.  
Es zeigt Schritt für Schritt, wie man aus Dokumenten einen durchsuchbaren Vektorstore baut, wie synthetische Datensätze entstehen und wie einfache und fortgeschrittene RAG-Systeme evaluiert werden können.

---

## 0. Voraussetzungen

Um das Repository lokal nutzen zu können, benötigt Ihr:

- **Docker**: [Installationsanleitung](https://docs.docker.com/get-docker/)  
- **Docker Compose** (wird oft mit Docker Desktop automatisch installiert)  
- **Git** (um dieses Repository zu klonen):  
```
  git clone git@github.com:sciencemediacenter/lab_scicar2025_rag_workshop.git
  cd lab_scicar2025_rag_workshop
```

## 1. Container starten

Das Projekt ist so vorbereitet, dass es direkt in einer Docker-Umgebung läuft.

1. Container bauen:
```
  docker-compose build
```

2. Container starten:
```
  docker-compose up
```

3. Mit VS Code (oder einer anderen IDE) könnt ihr euch an den laufenden Container anhängen und die Jupyter-Notebooks im Ordner `notebooks/` ausführen.

## 2. .env-Datei einrichten

Im Repository liegt eine Vorlage .env.example.
Ihr könnt euch daraus eure persönliche `.env` erstellen:
```
  cp .env.example .env
```

Dort müssen dann die API Keys der einzelnen Sprachmodelle eingefügt werden:
```
  #######################################
  #  LLM providers                      #
  #######################################
  OPENAI_API_KEY=
  ANTHROPIC_API_KEY=
  COHERE_API_KEY=
```

- OpenAI und Anthropic erfordern kostenpflichtige Accounts

- Cohere bietet einen kostenlosen Zugang mit Rate-Limit

Es reicht, einen Anbieter einzutragen. In diesem Fall (oder wenn ihr andere Modelle nutzen möchtet, z. B. Google) müsst ihr die dazugehörigen Codestellen anpassen.

## 3. Überblick über die Notebooks

Die wichtigsten Schritte laufen in den Jupyter-Notebooks im Ordner `notebooks/`:

##### 01_build_vectorstore.ipynb
Aus der Dokumentensammlung wird ein semantisch durchsuchbarer Vektorstore erstellt.

##### 02_create_synthetic_dataset.ipynb
Generiert synthetische Frage-Antwort-Paare, filtert für bessere Qualität und prüft Diversität.

##### 03_run_naive_rag.ipynb
Einfaches RAG-System mit den 3 Kernelementen (Retriever, LLM, Generator).
Testset wird damit ausgewertet, Metriken berechnet.

##### 04_run_advanced_rag.ipynb
Erweiterte Variante mit zusätzlichen Funktionen.
Ergebnisse werden erneut evaluiert.

##### 05_compare_rag_variants.ipynb
Vergleich der beiden Varianten, Ergebnisse werden grafisch dargestellt.

##### 06_use_own_metrics.ipynb
Exkurs: Beispiele für eigene Metriken und deren Einsatz.

## 4. Ergebnisse und Daten

`data/` 
→ Beispiel-Dokumente, Prompts, synthetische QA-Paare, Vektorstore

`results/` 
→ Ausgewertete Ergebnisse der RAG Experimente

`src/` 
→ Python-Code für Naive und Advanced RAG

`notebooks/` → Jupyter-Notebooks 

## 5. Slides zum Vortrag

Die Folien zum Vortrag liegen im Ordner `slides/`.