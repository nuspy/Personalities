"""Generazione delle conversazioni di addestramento a partire dalle domande.

Differenze rispetto alla versione precedente:

- le richieste all'LLM vanno **in parallelo**: erano sequenziali, e con un
  modello locale a 20 secondi per risposta mille esempi sono cinque ore;
- il parsing della risposta accetta le forme in cui i modelli rispondono
  davvero (oggetto, lista, oggetto con chiave contenitore, testo con
  delimitatori di codice) invece di restituire una lista vuota in silenzio;
- ogni conversazione porta la domanda, la categoria e il passo di origine,
  cosi' un esempio sbagliato si risale fino alla fonte.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..data_models import CompleteStyleProfile, TrainingConversation
from ..utils.llm_provider import LLMError, LLMProvider
from .question_generator import CATEGORY_ANACHRONISTIC, CATEGORY_HISTORICAL, Question

logger = logging.getLogger(__name__)

# Una risposta piu' corta di cosi' non e' un esempio utile.
MIN_ANSWER_CHARS = 40

# Contesto massimo passato al modello per una singola domanda.
MAX_CONTEXT_CHARS = 3000


class ConversationGenerator:
    """Trasforma le domande in conversazioni nello stile del personaggio."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.llm = llm_provider
        self.config = config or {}
        dataset_config = self.config.get("dataset", {})
        self.max_workers = max(1, int(dataset_config.get("max_workers", 4)))
        self._lock = threading.Lock()
        self.last_failures: Counter = Counter()
        # Attacchi gia' usati: servono a chiedere varieta' nelle richieste
        # successive. Le istruzioni statiche non bastano — un profilo che
        # elenca le formule tipiche induce il modello a ripeterle sempre, e
        # su un dataset misurato 17 risposte su 18 aprivano allo stesso modo.
        self._used_openings: Counter = Counter()
        self.opening_repetition_limit = int(
            dataset_config.get("max_same_opening", 3)
        )

    # ------------------------------------------------------------- pubblico

    def generate(
        self,
        profile: CompleteStyleProfile,
        questions: Sequence[Question],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[TrainingConversation]:
        """Genera una conversazione per domanda, in parallelo."""
        if not questions:
            return []

        conversations: List[TrainingConversation] = []
        completed = 0
        self._used_openings = Counter()
        # I motivi degli scarti si accumulano qui: se alla fine non resta
        # nulla, l'errore deve poter dire *perche'*, non limitarsi a
        # constatare il vuoto.
        self.last_failures = Counter()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._generate_one, profile, question): question
                for question in questions
            }

            for future in as_completed(futures):
                completed += 1
                try:
                    conversation = future.result()
                except Exception as exc:
                    self._record_failure(f"{type(exc).__name__}: {exc}")
                    conversation = None

                if conversation is not None:
                    conversations.append(conversation)

                if progress_callback:
                    progress_callback(completed, len(questions))

        self._log_diversity(conversations)

        failures = sum(self.last_failures.values())
        if failures:
            logger.warning(
                f"{failures}/{len(questions)} domande senza risposta utilizzabile. "
                f"Motivi: {self.failure_summary()}"
            )
        return conversations

    def _log_diversity(self, conversations: List[TrainingConversation]) -> None:
        """Segnala se le risposte si assomigliano troppo nell'attacco.

        Un dataset in cui quasi tutte le risposte iniziano allo stesso modo
        insegna quel tic e nient'altro: meglio saperlo prima di addestrare
        che dopo, guardando il modello ripetere la stessa formula.
        """
        if len(conversations) < 5:
            return

        openings = Counter(
            " ".join(c.turns[-1]["content"].strip().split()[:3]).lower()
            for c in conversations
        )
        most_common, count = openings.most_common(1)[0]
        share = count / len(conversations)
        if share >= 0.5:
            logger.warning(
                f"{count} risposte su {len(conversations)} ({share:.0%}) aprono con "
                f"«{most_common}»: il dataset insegnerebbe soprattutto questa ripetizione. "
                "Conviene un corpus piu' ampio o un numero maggiore di esempi."
            )

    def _record_failure(self, reason: str) -> None:
        """Registra il motivo di uno scarto, normalizzandolo per raggrupparlo."""
        with self._lock:
            self.last_failures[_normalize_reason(reason)] += 1

    def failure_summary(self, limit: int = 3) -> str:
        """I motivi di scarto piu' frequenti, in forma leggibile."""
        if not self.last_failures:
            return ""
        return "; ".join(
            f"{reason} ({count}x)"
            for reason, count in self.last_failures.most_common(limit)
        )

    # -------------------------------------------------------------- privato

    def _generate_one(
        self, profile: CompleteStyleProfile, question: Question
    ) -> Optional[TrainingConversation]:
        answer = self._ask(profile, question)
        if answer is None:
            return None

        # Chiedere varieta' nel prompt non basta: il profilo elenca le formule
        # dell'autore e il modello le considera prescrittive, tanto piu' quanto
        # piu' il corpus e' ristretto. Se l'attacco e' gia' saturo si rigenera
        # una volta sola, vietandolo esplicitamente — il ritentativo costa una
        # chiamata, e non vale la pena spenderne piu' di una per esempio.
        if self._opening_is_saturated(answer):
            retry = self._ask(profile, question, forbid_opening=_opening_of(answer))
            if retry is not None and not self._opening_is_saturated(retry):
                answer = retry

        self._record_opening(answer)

        return TrainingConversation(
            id=f"{question.category}_{abs(hash(question.text)) % 10**8:08d}",
            conversation_type=question.category,
            system_prompt=profile.generated_system_prompt,
            turns=[
                {"role": "user", "content": question.text},
                {"role": "assistant", "content": answer},
            ],
            metadata={
                "category": question.category,
                "source_segment_id": question.source_segment_id or "",
                "grounded": bool(question.context),
                **question.metadata,
            },
        )

    def _ask(
        self,
        profile: CompleteStyleProfile,
        question: Question,
        forbid_opening: str = "",
    ) -> Optional[str]:
        """Una richiesta al modello, con la risposta gia' validata."""
        prompt = self._build_prompt(profile, question, forbid_opening)

        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=(
                    "Interpreti un personaggio storico rispettandone rigorosamente "
                    "lo stile descritto. Rispondi solo con l'oggetto JSON richiesto."
                ),
                json_mode=True,
            )
        except LLMError as exc:
            self._record_failure(str(exc))
            return None

        answer = self._extract_answer(response)
        if not answer:
            self._record_failure("risposta vuota o non interpretabile")
            return None
        if len(answer) < MIN_ANSWER_CHARS:
            self._record_failure(f"risposta piu' corta di {MIN_ANSWER_CHARS} caratteri")
            return None
        return answer

    def _opening_is_saturated(self, answer: str) -> bool:
        with self._lock:
            return self._used_openings[_opening_of(answer)] >= self.opening_repetition_limit

    def _overused_openings(self) -> List[str]:
        """Attacchi gia' usati troppe volte, da non riproporre."""
        with self._lock:
            return [
                opening for opening, count in self._used_openings.items()
                if count >= self.opening_repetition_limit
            ][:6]

    def _record_opening(self, answer: str) -> None:
        opening = _opening_of(answer)
        if opening:
            with self._lock:
                self._used_openings[opening] += 1

    def _build_prompt(
        self,
        profile: CompleteStyleProfile,
        question: Question,
        forbid_opening: str = "",
    ) -> str:
        guidelines = profile.training_guidelines or {}
        do_rules = guidelines.get("do", [])[:6]
        avoid_rules = guidelines.get("avoid", [])[:6]

        parts = [
            f"PERSONAGGIO: {profile.author_name}",
        ]
        if profile.era:
            parts.append(f"EPOCA: {profile.era}")

        parts.append("\nCOME PARLA:\n" + profile.generated_system_prompt)

        if do_rules:
            parts.append("\nREGOLE DA RISPETTARE:\n" + "\n".join(f"- {r}" for r in do_rules))
        if avoid_rules:
            parts.append("\nDA EVITARE:\n" + "\n".join(f"- {r}" for r in avoid_rules))

        if question.context:
            # Il contesto ancora la risposta a un passo reale: senza, il
            # modello inventerebbe fatti plausibili ma non attestati.
            parts.append(
                "\nPASSO DI RIFERIMENTO (i fatti devono venire da qui, "
                "riformulati nello stile del personaggio, senza aggiungerne):\n"
                f'"""{question.context[:MAX_CONTEXT_CHARS]}"""'
            )

        if question.category == CATEGORY_ANACHRONISTIC:
            parts.append(
                "\nISTRUZIONE SPECIALE: la domanda riguarda qualcosa che il personaggio "
                "non puo' conoscere. Non deve fingere di capirla ne' spiegarla: "
                "deve non riconoscerla, chiedere chiarimenti o interpretarla con le "
                "categorie del proprio tempo, restando nel personaggio."
            )
        elif question.category == CATEGORY_HISTORICAL and not question.context:
            parts.append(
                "\nISTRUZIONE SPECIALE: rispondi solo su cio' di cui il personaggio ha "
                "esperienza diretta. Se non lo sa, deve dirlo."
            )

        parts.append(f"\nDOMANDA: {question.text}")
        parts.append(
            '\nRispondi con questo JSON esatto: {"risposta": "<la risposta del personaggio>"}'
        )

        return "\n".join(parts)

    # ---------------------------------------------------------------- parsing

    @staticmethod
    def _extract_answer(response: str) -> str:
        """Estrae la risposta dalle forme in cui i modelli rispondono davvero."""
        if not response:
            return ""

        cleaned = _strip_code_fence(response)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Nessun JSON valido: si cerca un oggetto annidato nel testo, e
            # in ultima istanza si accetta il testo cosi' com'e' — una
            # risposta in chiaro e' comunque utilizzabile.
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    return cleaned.strip()
            else:
                return cleaned.strip()

        return _first_text(data)


def _first_text(data: Any) -> str:
    """Primo valore testuale utile, qualunque forma abbia la risposta."""
    if isinstance(data, str):
        return data.strip()

    if isinstance(data, dict):
        # Chiavi attese, in ordine di preferenza.
        for key in ("risposta", "answer", "response", "content", "text", "output"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if value is not None:
                nested = _first_text(value)
                if nested:
                    return nested
        # Nessuna chiave nota: si prende il primo valore testuale sostanzioso.
        for value in data.values():
            nested = _first_text(value)
            if len(nested) >= MIN_ANSWER_CHARS:
                return nested
        return ""

    if isinstance(data, list):
        for item in data:
            nested = _first_text(item)
            if nested:
                return nested

    return ""


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _normalize_reason(reason: str) -> str:
    """Accorcia e normalizza un motivo di scarto, per poterlo raggruppare.

    I messaggi contengono numeri variabili (token consumati, codici) che
    renderebbero ogni occorrenza diversa dalle altre e impedirebbero di
    vedere che si tratta sempre dello stesso problema.
    """
    cleaned = re.sub(r"\d+", "N", reason)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:160]


def _opening_of(answer: str) -> str:
    """Le prime parole di una risposta, normalizzate per il confronto."""
    return " ".join(answer.strip().split()[:3]).lower().strip("«»\"'")
