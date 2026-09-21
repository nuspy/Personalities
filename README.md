# Historical Persona Pipeline

Costruisce la **personalità di un personaggio** — vocabolario, modo di ragionare,
espressioni tipiche, modo di parlare, valori — a partire da documenti, libri e
fonti online, e ne ricava un prompt di sistema e un dataset di addestramento LoRA.

Pensata per funzionare **senza intervento manuale**: basta un nome. I documenti
propri restano sempre caricabili, come aggiunta e non come requisito.

---

## Installazione

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e .
```

Dipendenze opzionali:

```bash
pip install -e ".[ancient]"   # latino e greco antico (CLTK, installazione pesante)
pip install -e ".[ocr]"       # PDF scansionati (richiede anche Tesseract nel PATH)
pip install -e ".[training]"  # addestramento e conversione (richiede GPU NVIDIA)
pip install -e ".[unsloth]"   # acceleratore opzionale per l'addestramento
pip install -e ".[gguf]"      # dipendenze Python dell'export GGUF
pip install -e ".[dev]"       # test
```

**GPU recenti (RTX 50xx, Blackwell)**: PyTorch va installato con CUDA 12.8 o
superiore, altrimenti la GPU non viene riconosciuta.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

`unsloth` è facoltativo: se non è installabile — succede spesso su Windows e
sulle GPU più recenti — l'addestramento prosegue con transformers + peft, che
è il percorso su cui il ciclo è stato verificato.

I modelli linguistici si scaricano da soli alla prima esecuzione (versione
piccola, qualche decina di MB). Per una qualità superiore:

```bash
python -m spacy download it_core_news_lg
```

> **Rete aziendale o antivirus con ispezione TLS**: il pacchetto `truststore`
> è già fra le dipendenze e aggancia la verifica dei certificati a quella del
> sistema. Senza, ogni richiesta fallisce con `CERTIFICATE_VERIFY_FAILED`
> anche se il browser sulla stessa macchina funziona.

---

## Uso

### Automatico, partendo dal solo nome

```bash
persona build "Giulio Cesare" --era "Repubblica romana, I secolo a.C."
```

Cerca le fonti online, scarta gli omonimi, analizza il corpus, costruisce il
profilo e genera il dataset.

### Con documenti propri

```bash
persona build "Marco Aurelio" --files ./corpus --no-research
```

`--files` accetta file singoli, cartelle (esplorate ricorsivamente) e pattern.

### Fermarsi prima

```bash
persona build "Cicerone" --until analyze     # solo il profilo, nessun LLM richiesto
```

Stadi: `research` → `ingest` → `analyze` → `dataset` → `train`.

### Altri comandi

```bash
persona list                       # progetti esistenti
persona inspect ./progetti/cesare  # profilo completo
persona inspect ./progetti/cesare --prompt-only > prompt.md
```

### Addestramento

Quattro modalità, che differiscono per **da dove partono i pesi**:

| Modalità | Cosa fa |
|---|---|
| `lora_new` | Adapter nuovo sul modello base. È il punto di partenza. |
| `lora_continue` | Riprende un adapter esistente e prosegue da lì. L'adapter **viene modificato**: conviene copiarlo se serve tornare indietro. |
| `lora_stack` | Fonde uno o più adapter nel modello base e addestra un adapter nuovo sopra. Gli originali restano intatti. |
| `full_finetune` | Addestra tutti i pesi. Qualità superiore, molta più VRAM, e il risultato è un modello intero anziché pochi MB. |

```bash
# nuova LoRA (default)
persona train ./progetti/cesare

# continua una LoRA esistente con altro materiale
persona train ./progetti/cesare --mode lora_continue \
    --from-adapter ./progetti/cesare/output/lora_adapters

# LoRA della persona sopra una LoRA di stile già addestrata
persona train ./progetti/cesare --mode lora_stack \
    --from-adapter ./lora/latino ./lora/retorica \
    --adapter-weight 0.7 0.3 \
    --output-name cesare_v2

# elenca gli adapter disponibili
persona adapters
```

Indicare `--from-adapter` senza `--mode` implica `lora_stack`: continuare è la
scelta più specifica e va dichiarata.

**Fusione di più adapter** — `linear` (default) fa la media pesata e richiede
lo stesso rango; `svd` è l'unica che gestisce ranghi diversi; `ties` e
`dare_ties` riducono l'interferenza fra adapter addestrati su compiti diversi;
`cat` li concatena sommando i ranghi.

La compatibilità è verificata **prima** di caricare il modello: adapter di
modelli base diversi producono pesi incoerenti, non un errore, quindi vengono
bloccati subito.

### Conversione

Un adapter da solo non si usa: pesa pochi MB ma richiede il modello base a
fianco. La conversione lo rende eseguibile altrove.

```bash
# modello autonomo a 16 bit
persona convert ./progetti/cesare/output/lora_adapters --format merged_16bit

# GGUF quantizzato per LM Studio / llama.cpp
persona convert ./progetti/cesare/output/lora_adapters \
    --format gguf --quantization q4_k_m

# pacchetto Ollama col prompt del personaggio già dentro
persona convert ./progetti/cesare/output/lora_adapters \
    --format ollama --quantization q4_k_m --project ./progetti/cesare

# fonde due adapter in uno solo, senza addestrare
persona convert ./lora/a ./lora/b --format adapter_only \
    --weights 0.6 0.4 --merge-strategy ties
```

| Formato | Quando |
|---|---|
| `merged_16bit` | Modello autonomo, qualità piena. ~2 GB per miliardo di parametri. |
| `merged_4bit` | Un quarto dello spazio, perdita contenuta. |
| `adapter_only` | Solo l'adapter (pochi MB), utile per archiviare o combinare. |
| `gguf` | File unico per llama.cpp, LM Studio, Jan. CPU o GPU. |
| `ollama` | GGUF + Modelfile con il prompt di sistema incorporato. |

Quantizzazioni GGUF: `f16`, `q8_0`, `q6_k`, `q5_k_m`, **`q4_k_m`** (default),
`q3_k_m`, `q2_k`. Lo spazio stimato è mostrato prima di iniziare — la
differenza fra `q4_k_m` e `f16` di un 8B è di una decina di gigabyte, e a
disco pieno ci si accorge solo a metà conversione.

> **GGUF richiede llama.cpp**, non installabile via pip:
> ```bash
> git clone https://github.com/ggerganov/llama.cpp
> cmake -B build llama.cpp && cmake --build build --config Release
> ```
> Indicarne il percorso in `conversion.llama_cpp_path` o nella variabile
> d'ambiente `LLAMA_CPP_PATH`. Gli altri formati non ne hanno bisogno.
>
> Chi ha **Unsloth Studio** ha gia' una build in `~/.unsloth/llama.cpp`, che
> viene trovata da sola.

### Interfaccia grafica

```bash
python -m historical_persona_pipeline.main
```

Quattro tab: **Setup** (file e ricerca), **Elaborazione** (la pipeline),
**Addestramento** (scelta della modalità e degli adapter di partenza),
**Conversione** (formato e quantizzazione). Il tab Conversione è indipendente:
si può aprire il programma solo per convertire un adapter già esistente.

---

## Cosa produce

```
progetti/<nome>/
├── sources/online/          # documenti scaricati + manifest con le fonti
├── ingestion.json           # corpus segmentato
├── profiles/
│   ├── style_profile.json   # tutte le misure
│   └── system_prompt.md     # il prompt da usare con qualsiasi modello
├── datasets/
│   ├── raw_dataset.json
│   ├── train_dataset.jsonl  # formato ShareGPT
│   └── val_dataset.jsonl
├── output/<nome>/           # adapter addestrato
│   ├── adapter_model.safetensors
│   └── persona_training.json    # modalità, base, adapter di provenienza
└── exports/<formato>_<data>/    # risultato della conversione
    └── conversion_manifest.json
```

Ogni adapter porta con sé `persona_training.json`: su quale modello base è
stato addestrato e da quali adapter deriva. Senza, una cartella di pesi è
indistinguibile da un'altra e la catena delle fusioni diventa irricostruibile.

Il **profilo si costruisce senza LLM**: `--until analyze` funziona completamente
offline. L'endpoint LLM serve solo per generare il dataset.

---

## Cosa viene misurato

| Dimensione | Come |
|---|---|
| **Vocabolario** | lemmi ordinati per frequenza *ponderata dalla diffusione*: un termine ripetuto in una sola pagina non è caratteristico |
| **Espressioni** | n-grammi ricorrenti — le formule che l'autore ripete (*«quibus rebus cognitis»*) |
| **Modo di parlare** | incipit di frase, vocativi, tasso di domande, incisi, registro, modalità (certezza / dubbio / dovere) |
| **Modo di ragionare** | connettivi argomentativi in 9 lingue: causale, contrasto, concessione, autorità, esempio, condizione, enumerazione |
| **Valori** | dizionari con corrispondenza morfologica, ancorata ai confini di parola |
| **Sintassi** | lunghezza del periodo, subordinazione, coordinazione, persona verbale |
| **Prospettiva** | prima persona, terza persona, o terza persona riferita a sé — il tratto più riconoscibile di una voce |

### Il profilo tace su ciò che non può stabilire

Due casi in cui una misura non viene riportata, invece di essere riportata
come incerta:

- **modello morfologico assente** — subordinazione e coordinazione valgono 0
  perché non misurabili, non perché nulle;
- **corpus troppo piccolo** — sotto 120 frasi le aperture caratteristiche e
  sotto 5 000 token le formule non sono distinguibili dalle ripetizioni del
  campione. Un'apertura presente in oltre metà delle frasi viene esclusa
  comunque: descrive un corpus omogeneo, non uno stile.

Non è prudenza formale. Su un corpus di 234 parole *«quibus»* compariva in 3
frasi su 8 e finiva nel profilo come apertura caratteristica; il modello la
prendeva per prescrizione e apriva così **18 risposte su 18**. Tre tentativi
di correggere il problema a valle — linee guida più caute, richiesta esplicita
di varietà, rigenerazione con divieto dell'attacco — non hanno spostato nulla.
Nessuna istruzione corregge un profilo che afferma il falso con autorevolezza:
chi lo legge, modello compreso, lo prende alla lettera.

---

## Formati supportati

`.txt` `.md` `.pdf` `.docx` `.doc` `.rtf` `.odt` `.epub` `.html` `.xml/.tei`

Un file con estensione sconosciuta ma contenuto testuale viene letto comunque.

Dettagli che contano sui corpora reali:

- **PDF** — de-sillabazione di fine riga, rimozione di testatine e numeri di
  pagina ricorrenti, OCR sui documenti scansionati (se Tesseract è installato).
  Un PDF senza testo estraibile produce un **errore esplicito**, non un corpus
  vuoto silenzioso.
- **Codifiche** — UTF-8, rilevamento automatico, latin-1 come ultima risorsa,
  sempre con segnalazione.
- **File grandi** — il testo è suddiviso in blocchi e l'analisi NLP procede a
  lotti con un tetto di memoria configurabile: un libro intero non satura la RAM.

---

## Ricerca online

Tre fonti senza chiave API: **Wikipedia** (contesto), **Wikisource** (testi
originali), **Project Gutenberg** (opere integrali).

Le richieste rispettano il limite di frequenza dei servizi: pausa fra le
chiamate, `Retry-After` onorato, rallentamento progressivo dopo un 429.

I risultati passano da un **filtro di pertinenza** che combina corrispondenza
del titolo, densità del nome e coerenza temporale. Una ricerca per "Giulio
Cesare" restituisce anche *Augusto*, *Germanico Giulio Cesare* e *Giulio Cesare
Vanini*, filosofo del Seicento: senza filtro il profilo misurerebbe la media fra
persone diverse. Gli scarti restano tracciati nel manifest con il motivo.

Soglia regolabile con `research.relevance_threshold` (default `0.40`).

**Limite noto**: i parenti stretti con lo stesso nome e la stessa epoca
(*Germanico Giulio Cesare*) restano il caso più difficile — si trovano appena
sotto la soglia. Verificare il manifest quando il personaggio appartiene a una
dinastia.

---

## Generazione delle domande

Copre le sei categorie configurate in `dataset.conversation_types`:
fatti storici, valori, anacronismi, personalità, competenza specialistica,
conversazione quotidiana.

Funziona in due modi:

- **estrattivo** — le domande nascono dal corpus (entità nominate, valori
  misurati, campi semantici). Nessuna rete, nessun LLM;
- **generativo** — se un LLM è raggiungibile, le domande vengono riformulate in
  modo più naturale. Se non risponde, restano quelle estrattive.

Le domande sui fatti storici portano con sé il passo da cui nascono: la
risposta si àncora a quel testo invece di inventare fatti plausibili.

---

## Configurazione

`historical_persona_pipeline/config/default_config.yaml`. Ogni chiave presente è
effettivamente letta; i valori mancanti sono completati dai default, quindi un
file parziale non impedisce l'avvio.

Parametri che conviene conoscere:

| Chiave | Effetto |
|---|---|
| `research.relevance_threshold` | severità del filtro sugli omonimi |
| `ingestion.target_chunk_chars` | dimensione dei blocchi di testo |
| `analysis.max_chars_per_nlp_batch` | tetto di memoria dell'analisi |
| `dataset.llm_base_url` | endpoint OpenAI-compatibile (LM Studio, Ollama, vLLM, Unsloth Studio…) |
| `dataset.max_workers` | richieste LLM in parallelo |
| `dataset.conversation_types` | proporzioni fra le categorie di domande |
| `training.mode` | `lora_new` · `lora_continue` · `lora_stack` · `full_finetune` |
| `training.start_from_adapters` | adapter di partenza, con peso opzionale |
| `training.merge_strategy` | come fondere più adapter |
| `training.max_steps` | `-1` addestra per `num_epochs`; un valore positivo lo sovrascrive |
| `conversion.llama_cpp_path` | percorso di llama.cpp per l'export GGUF |
| `conversion.keep_intermediate` | conserva il fp16 intermedio (decine di GB) |

---

## Il provider LLM

Serve solo allo stadio 3. Qualsiasi endpoint OpenAI-compatibile va bene:

```yaml
dataset:
  llm_provider: "lm_studio"          # o openai_compatible, ollama, vllm, anthropic
  llm_base_url: "http://127.0.0.1:1234/v1"
  llm_model_name: "<nome del modello caricato>"
  llm_api_key: "lm-studio"
```

**La modalità JSON viene negoziata da sola.** I server non concordano su come
si chiede una risposta JSON: OpenAI e la maggior parte dei cloni usano
`response_format: json_object`, LM Studio recente accetta solo `json_schema` o
`text`, altri non supportano nulla. Il provider prova in ordine, ricorda cosa
funziona e, se nessun vincolo è accettato, prosegue senza: il prompt chiede
comunque JSON e il parser tollera il testo libero.

Quando una richiesta viene rifiutata, il messaggio d'errore riporta **il corpo
della risposta del server**: un 400 senza quello costringe a indovinare.

---

## Test

```bash
pytest historical_persona_pipeline/tests -q
```

I test non toccano la rete né caricano modelli. Ogni test corrisponde a un
difetto reale trovato nel codice: il nome dice quale comportamento sbagliato
non deve tornare.

### Cosa è stato verificato eseguendolo davvero

Il ciclo e' stato eseguito **due volte**, su entrambi i percorsi di
caricamento: con `transformers + peft` e con **Unsloth 2026.9** attivo.
RTX 5090, PyTorch 2.11+cu128, transformers 4.57, TRL 0.23, PEFT 0.21,
`Qwen2.5-0.5B-Instruct` — 10 fasi su 10:

| Fase | Esito |
|---|---|
| `lora_new` | adapter prodotto, loss 3,23 → 3,02 |
| `lora_continue` | adapter ripreso e proseguito |
| `lora_stack` | due adapter fusi (0,7/0,3) e nuovo adapter addestrato sopra |
| `full_finetune` | tutti i pesi addestrati |
| `merged_16bit` | modello autonomo salvato |
| `adapter_only` | due adapter fusi in uno senza addestrare |
| `merged_4bit` | quantizzato con bitsandbytes |
| fusione `ties` | strategia alternativa a `linear` |
| `gguf` q4_k_m | 994 MB f16 → 295 MB quantizzato, caricato da `llama-cli` a 47 tok/s |
| Modelfile Ollama | prompt di sistema incorporato |

Un addestramento più lungo (80 step, loss finale 0,18) conferma che l'adapter
**modifica davvero** il comportamento del modello: stesso prompt, risposta
diversa dal modello base.

### Difetti che solo l'esecuzione ha rivelato

Nessuno di questi produceva un errore al momento in cui nasceva.

**Le conversazioni svuotate.** I template di chat leggono `role`/`content`,
il dataset e' in ShareGPT (`from`/`value`). Passandoglielo cosi' il template
scarta l'intera conversazione e restituisce il prompt di default del modello.
Quattro addestramenti erano terminati "con successo", loss in discesa, su
esempi privi di contenuto. Il parametro `mapping` di Unsloth sembra fatto
apposta per evitarlo, ma **non converte i dati**: tiene solo i turni il cui
ruolo coincide gia' con lo standard (`system`) e scarta `human` e `gpt`. Ora
la conversione e' esplicita e incondizionata, e un controllo dopo la
formattazione verifica che il testo contenga davvero le conversazioni.

**Il modello "fuso" che non lo era.** `save_pretrained_merged` applicato a un
modello avvolto da PEFT lascia nei pesi la struttura dell'adapter
(`...q_proj.base_layer.weight`). Il salvataggio riesce, il modello si carica,
e il problema emerge molto dopo: la conversione GGUF si ferma su un tensore
sconosciuto. Ora si usa sempre `merge_and_unload` e i pesi salvati vengono
ispezionati per escludere residui PEFT.

**Il gradiente interrotto.** In `lora_continue` con gradient checkpointing gli
input embedding devono propagare il gradiente, altrimenti il grafo si spezza:
l'errore, «element 0 of tensors does not require grad», non nomina ne' gli
adapter ne' il checkpointing.

**L'ordine degli import.** `truststore` sostituisce la classe di contesto TLS,
ma non tocca le sessioni HTTPS gia' costruite. Importando Unsloth prima del
package, `huggingface_hub` nasce con i certificati vecchi e i download dei
modelli falliscono. Il vincolo e' documentato in `__init__.py`.

Quando Unsloth non riesce a caricare un modello — capita, ricompila i moduli e
la compilazione dipende da combinazioni di versioni — l'addestramento **non si
ferma**: ripiega su transformers dichiarandolo.
