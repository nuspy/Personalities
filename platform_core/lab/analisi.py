"""Come sta andando: i numeri della console.

Ogni numero si calcola da ciò che il sistema registra già — messaggi, tracce
delle risposte, registro dei crediti, voti — e non da contatori tenuti a
parte: un contatore che si aggiorna in un punto e si dimentica in un altro
racconta una storia diversa da quella del database, e nessuno sa quale sia
vera.

Ogni indicatore arriva col valore del periodo precedente di pari durata:
«1.284 risposte» non dice se è tanto o poco, «1.284 contro 950» sì.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..domain.base import utcnow


def _intervallo(giorni: int, fine: Optional[datetime] = None) -> tuple[datetime, datetime]:
    fine = fine or utcnow()
    # Il giorno corrente incluso per intero: i numeri di oggi crescono
    # mentre si guardano, ed è ciò che ci si aspetta da una console.
    fine = (fine + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return fine - timedelta(days=giorni), fine


_INDICATORI = text("""
    with risposte as (
        select m.id, c.owner_id
          from messages m
          join conversations c on c.id = m.conversation_id
         where m.role = 'assistant' and m.created_at >= :dal and m.created_at < :al
    ),
    domande as (
        select distinct c.owner_id
          from messages m
          join conversations c on c.id = m.conversation_id
         where m.role = 'user' and m.created_at >= :dal and m.created_at < :al
    ),
    tracce as (
        select t.*
          from answer_traces t
         where t.created_at >= :dal and t.created_at < :al
    )
    select
        (select count(*) from risposte) as risposte,
        (select count(*) from domande) as utenti_attivi,
        (select count(*) from conversations
          where created_at >= :dal and created_at < :al) as conversazioni_nuove,
        (select coalesce(sum(-delta) filter (where reason = 'consumo'), 0)
              - coalesce(sum(delta) filter (where reason = 'rimborso'), 0)
           from credit_ledger
          where created_at >= :dal and created_at < :al) as crediti,
        (select count(*) from answer_feedback
          where vote = 1 and updated_at >= :dal and updated_at < :al) as su,
        (select count(*) from answer_feedback
          where vote = -1 and updated_at >= :dal and updated_at < :al) as giu,
        (select count(*) from tracce) as tracce,
        (select count(*) from tracce
          where jsonb_array_length(coalesce(grounding->'riferimenti_inventati', '[]'::jsonb)) > 0
        ) as con_inventati,
        (select count(*) from tracce where grounding->>'livello' = 'nli') as verificate,
        (select count(*) from tracce
          where grounding->>'livello' = 'nli' and (grounding->>'fondata')::boolean
        ) as fondate,
        (select percentile_cont(0.5) within group (order by (latency_ms->>'primo_token')::float)
           from tracce where latency_ms->>'primo_token' is not null) as primo_token_p50,
        (select percentile_cont(0.95) within group (order by (latency_ms->>'primo_token')::float)
           from tracce where latency_ms->>'primo_token' is not null) as primo_token_p95,
        (select coalesce(sum((usage->>'prompt_tokens')::bigint), 0) from tracce) as token_prompt,
        (select coalesce(sum((usage->>'cached_tokens')::bigint), 0) from tracce) as token_da_cache
""")

_AL_GIORNO = text("""
    select date_trunc('day', m.created_at at time zone 'UTC') as giorno,
           count(*) filter (where m.role = 'assistant') as risposte,
           count(distinct c.owner_id) filter (where m.role = 'user') as utenti
      from messages m
      join conversations c on c.id = m.conversation_id
     where m.created_at >= :dal and m.created_at < :al
     group by 1
     order by 1
""")

_PER_PERSONALITA = text("""
    select p.slug, p.display_name as nome,
           count(m.id) filter (where m.role = 'assistant') as risposte,
           count(distinct c.owner_id) as utenti,
           count(f.message_id) filter (where f.vote = 1) as su,
           count(f.message_id) filter (where f.vote = -1) as giu
      from messages m
      join conversations c on c.id = m.conversation_id
      join personalities p on p.id = c.personality_id
      left join answer_feedback f on f.message_id = m.id
     where m.created_at >= :dal and m.created_at < :al
     group by p.slug, p.display_name
     order by risposte desc
""")


def _quota(parte: int, totale: int) -> Optional[float]:
    return parte / totale if totale else None


def _indicatori(riga) -> Dict[str, Any]:
    su, giu = int(riga.su), int(riga.giu)
    return {
        "risposte": int(riga.risposte),
        "utenti_attivi": int(riga.utenti_attivi),
        "conversazioni_nuove": int(riga.conversazioni_nuove),
        "crediti": int(riga.crediti),
        "voti": {"su": su, "giu": giu, "approvazione": _quota(su, su + giu)},
        "citazioni_inventate": _quota(int(riga.con_inventati), int(riga.tracce)),
        "verifica": {
            "verificate": int(riga.verificate),
            "fondate": int(riga.fondate),
            "quota": _quota(int(riga.fondate), int(riga.verificate)),
        },
        "primo_token_ms": {
            "p50": round(riga.primo_token_p50) if riga.primo_token_p50 is not None else None,
            "p95": round(riga.primo_token_p95) if riga.primo_token_p95 is not None else None,
        },
        "cache": {
            "token_prompt": int(riga.token_prompt),
            "token_da_cache": int(riga.token_da_cache),
            "quota": _quota(int(riga.token_da_cache), int(riga.token_prompt)),
        },
    }


async def riepilogo(session: AsyncSession, *, giorni: int, fine: Optional[datetime] = None) -> Dict[str, Any]:
    dal, al = _intervallo(giorni, fine)
    prima_dal, prima_al = dal - timedelta(days=giorni), dal

    attuale = (await session.execute(_INDICATORI, {"dal": dal, "al": al})).one()
    precedente = (await session.execute(_INDICATORI, {"dal": prima_dal, "al": prima_al})).one()

    righe = {
        r.giorno.date(): r
        for r in (await session.execute(_AL_GIORNO, {"dal": dal, "al": al})).all()
    }
    # Tutti i giorni, anche quelli vuoti: un grafico che salta i giorni senza
    # attività li fa sparire, e un calo a zero sembra una linea continua.
    al_giorno: List[Dict[str, Any]] = []
    for n in range(giorni):
        giorno = (dal + timedelta(days=n)).date()
        r = righe.get(giorno)
        al_giorno.append({
            "giorno": giorno.isoformat(),
            "risposte": int(r.risposte) if r else 0,
            "utenti": int(r.utenti) if r else 0,
        })

    per_personalita = [
        {
            "slug": r.slug, "nome": r.nome,
            "risposte": int(r.risposte), "utenti": int(r.utenti),
            "su": int(r.su), "giu": int(r.giu),
            "approvazione": _quota(int(r.su), int(r.su) + int(r.giu)),
        }
        for r in (await session.execute(_PER_PERSONALITA, {"dal": dal, "al": al})).all()
    ]

    return {
        "periodo": {"dal": dal.isoformat(), "al": al.isoformat(), "giorni": giorni},
        "indicatori": _indicatori(attuale),
        "precedente": _indicatori(precedente),
        "al_giorno": al_giorno,
        "per_personalita": per_personalita,
    }
