# 1. Platform Architecture

## 1.1 Multi-user architecture
- Implement full multi-user support from the beginning.
- Separate frontend and backend applications/services.
- Strict user/session isolation.
- Each user has:
  - account
  - subscription
  - credit/response balance
  - chat history
  - short-term memories
  - long-term memories
  - personality-specific preferences and data
  - usage statistics
- All user-owned data must be associated with a tenant/user identity and protected by backend authorization.

## 1.2 Frontend / Backend separation
### Frontend
Responsible for:
- user interface
- chat rendering
- avatar rendering
- audio capture/playback
- client-side STT/TTS where supported
- client-side avatar animation
- local UI state
- streaming response visualization

### Backend
Responsible for:
- authentication
- authorization
- users
- subscriptions
- credits/quotas
- personality configuration
- RAG/CAG
- memory
- LLM orchestration
- provider management
- guardrails
- ground-check
- chat persistence
- analytics
- administration
- billing
- audit logging

**Important architectural rule:** avatar rendering and animation should remain client-side wherever technically possible. The backend should return the data necessary to animate the avatar, not render video frames.

---

# 2. Authentication & Account Management

## 2.1 Registration and login
Support:
- email/password
- Google
- Facebook
- Apple

Google/Facebook/Apple can initially be mocked, but the authentication architecture should already support real OAuth/OIDC providers.

## 2.2 Account management
- profile
- email verification
- password reset
- session management
- logout from individual/all devices
- account deletion
- export personal data
- language/preferences
- notification preferences

## 2.3 Authorization / RBAC
At minimum:
- User
- Administrator

Architecture should allow future roles such as:
- Super Admin
- Content Manager
- Support
- Analyst
- Billing Manager

---

# 3. Personality Management

## 3.1 Personality entity

Each personality should have:

- unique ID
- name
- display name
- description
- avatar
- system/personality prompt
- personality characteristics
- behavioral rules
- language configuration
- commercial category
- one or more personality types
- RAG configuration
- CAG configuration
- memory configuration
- guardrails
- ground-check configuration
- LLM configuration
- TTS configuration
- STT configuration
- availability status
- ordering/visibility
- metadata

## 3.2 Personality types

A personality may belong to **multiple types simultaneously**.

Examples:
- Historical figure
- Philosopher
- Politician
- Scientist
- Nobel Prize winner
- Writer
- Artist
- Military figure
- Entrepreneur
- Criminal
- Fictional character
- Celebrity
- Expert
- Coach
- Other

Types are created and managed by administrators.

This is separate from the **commercial category**.

### Example

**Julius Caesar**
- Commercial category: Gold
- Types:
  - Historical Figure
  - Military Figure
  - Politician

This distinction is important because commercial access and semantic classification serve completely different purposes.

## 3.3 Commercial personality categories

Introduce commercial access levels, for example:

- Free
- Base
- Gold
- Premium

These categories determine which personalities a subscription can access.

Administrators must be able to:
- create categories
- rename categories
- change ordering
- configure access rules
- associate personalities with categories

Normal users **cannot create personalities initially**.

---

# 4. Personality Administration

Dedicated Admin GUI for:

- create personality
- edit personality
- clone personality
- enable/disable personality
- archive personality
- assign commercial category
- assign multiple personality types
- configure prompt
- configure RAG
- configure CAG
- configure guardrails
- configure ground-check
- configure LLM
- configure TTS
- configure STT
- configure avatar
- configure memory behavior
- configure language availability

## 4.1 Personality versioning

Recommended addition:

Every personality should have versions.

Example:

`Julius Caesar v1.2`

Store:
- prompt version
- configuration version
- RAG configuration
- guardrail configuration
- model configuration
- publication date
- author/admin
- change history

This becomes extremely useful for debugging conversations and reproducing historical responses.

---

# 5. RAG / CAG Management

## 5.1 RAG management

Administrators can:

- create RAG sources
- edit RAG sources
- delete RAG sources
- enable/disable sources
- associate RAG sources with personalities
- configure retrieval parameters
- inspect indexed documents
- rebuild indexes
- monitor indexing status
- inspect retrieval results

Potential RAG sources:
- documents
- web sources
- structured data
- databases
- personality knowledge
- historical/reference material

## 5.2 CAG management

Same CRUD and assignment capabilities for CAG/context sources.

The system should explicitly distinguish:

**RAG**
→ external/retrieved knowledge

**CAG**
→ controlled/contextual information injected into the generation pipeline.

## 5.3 Retrieval observability

For administrators:
- query
- retrieved documents
- similarity/relevance scores
- reranking
- selected context
- rejected context
- retrieval latency
- retrieval failures

This will be important when debugging personality responses.

---

# 6. LLM Provider Management

Admin GUI for LLM providers.

## 6.1 Provider management
CRUD for:
- provider
- API credentials
- endpoint
- model
- capabilities
- pricing
- limits
- priority
- status

Support architecture for:
- cloud providers
- OpenAI-compatible APIs
- local models
- self-hosted inference
- multiple models per provider

## 6.2 Model routing

Configure:

- default model
- fallback model
- personality-specific model
- user/subscription-specific model
- model by task
- model by language
- model by latency requirement

Potential future routing:

`cheap model → normal conversation`

`premium model → premium personality`

`large model → complex reasoning`

## 6.3 Provider health

Track:
- availability
- latency
- error rate
- timeout rate
- tokens/sec
- input tokens
- output tokens
- cost
- rate limits
- current load

---

# 7. Guardrails & Ground-Check

## 7.1 Guardrails

Guardrails must be configurable per personality.

Support:
- input filtering
- output filtering
- topic restrictions
- prohibited content
- prompt injection detection
- jailbreak detection
- PII detection
- sensitive-data filtering
- hallucination controls
- response validation

## 7.2 Ground-check

Implement a separate verification layer that can evaluate:

- factual claims
- retrieved evidence
- source consistency
- unsupported claims
- contradictions
- confidence
- citation/evidence availability

Admin must be able to configure whether ground-check is:

- disabled
- optional
- mandatory

and at which stages it runs.

---

# 8. User Dashboard

The normal user interface should be completely separate from the Admin GUI.

## 8.1 Personality discovery

Users can browse personalities by:

- commercial category
- personality type
- language
- name
- popularity
- recently added
- recommended
- available/unavailable

Filters:
- Free / Base / Gold / Premium
- Historical
- Philosopher
- Politician
- Scientist
- etc.
- language

## 8.2 Personality page

Display:
- avatar
- name
- description
- personality types
- commercial category
- available languages
- short introduction
- capabilities
- remaining user access/credits

Actions:
- Start chat
- Continue previous chat
- View history

---

# 9. Chat System

## 9.1 Chat

Each chat initially uses **one personality**.

Features:
- streaming responses
- text input
- voice input
- voice output
- typing indicator
- generation status
- stop generation
- regenerate response
- edit user message
- retry
- copy
- share/export

## 9.2 Conversation history

Users can:
- browse conversations
- search conversations
- rename conversations
- archive conversations
- delete conversations
- export conversations

Export formats:
- TXT
- Markdown
- JSON
- PDF/HTML as future additions

## 9.3 Cross-personality memory reference

A user should be able to reference information previously discussed with another personality.

Example:

> "What do you think about my promotion plan that I discussed with @Julius_Caesar?"

The system should resolve:
1. current user
2. referenced personality
3. relevant previous conversation
4. relevant memory
5. temporal context

The current personality can then use the referenced memory within its own context.

This should **not** mean that all personalities automatically see all user data.

Memory sharing should be controlled by explicit policy/configuration.

---

# 10. Memory System

Memory is a first-class subsystem rather than simply part of chat history.

## 10.1 Short-term memory

Stores:
- current conversation context
- recent messages
- active topics
- current goals
- unresolved references
- recent entities

## 10.2 Long-term memory

Stores persistent user-specific information such as:
- preferences
- goals
- important events
- recurring topics
- relationships
- projects
- decisions
- personal facts explicitly worth remembering

## 10.3 Temporal memory

Every memory should have temporal metadata.

At minimum:

- created_at
- first_seen_at
- last_referenced_at
- last_updated_at
- source conversation
- source personality
- confidence
- relevance
- importance
- expiration/retention policy

The system should understand:

> "When did I last talk about X?"

and:

> "What did I tell you about X three weeks ago?"

## 10.4 Automatic memory retrieval

During a conversation the system should automatically retrieve relevant memories based on:

- semantic similarity
- recency
- importance
- frequency
- current topic
- personality relevance
- temporal references

This should use a hybrid retrieval strategy rather than pure vector similarity.

Potential scoring:

`memory_score = semantic_relevance + recency + importance + frequency + context_relevance`

## 10.5 Memory consolidation

Recommended addition:

Convert repeated short-term observations into durable long-term memories.

Example:

Several conversations:

> "I am looking for a new job."

Later:

> "I'm interviewing with company X."

The system can consolidate these into:

> User is currently looking for a new job and is evaluating company X.

This avoids uncontrolled memory growth.

## 10.6 Memory conflict resolution

Recommended addition.

If the user says:

> "I live in Budapest."

and later:

> "I moved to Vienna."

the memory subsystem should not blindly maintain both as current facts.

Maintain:
- historical value
- current value
- timestamps
- confidence
- superseded relationship

## 10.7 User memory controls

User can access a dedicated memory UI.

Actions:
- browse memories
- search memories
- inspect memory source
- edit memory
- delete individual memory
- delete all memories associated with a personality
- delete all memories
- disable automatic memory creation

Memory should be displayed in understandable form rather than exposing raw embeddings/vector data.

---

# 11. Admin Memory Management

Administrators require an advanced memory browser.

Features:

- search by user
- search by personality
- search by date
- search by memory type
- search by importance
- search by confidence
- inspect source conversation
- inspect related memories
- inspect memory history
- export memories
- audit memory creation/update/deletion

Admin access to user memories must be fully audited.

---

# 12. Chat & Memory Administration

Admin GUI should allow:

### User → Personality browsing

`User → Personality → Conversations → Messages → Memories`

and reverse navigation:

`Personality → Users → Conversations → Memories`

Functions:
- search
- filtering
- pagination
- export
- date filtering
- full-text search
- memory search
- conversation inspection

Recommended addition:

**Correlation ID / Conversation ID**

Every request should be traceable across:

`User → Chat → LLM → RAG → Guardrail → Ground-check → Memory → TTS`

This dramatically simplifies debugging.

---

# 13. Avatar System

## 13.1 Client-side rendering

Avatar rendering should happen in the browser.

The server should **not generate avatar video frames** for the normal interaction path.

Benefits:
- zero server GPU cost for avatar animation
- lower latency
- scalable architecture
- less bandwidth
- better concurrent-user capacity

## 13.2 Avatar types

Support initially:

### A. Static image + animation
- idle animation
- blinking
- subtle head movement
- mouth animation
- breathing/parallax effects

### B. Loop video
- idle video
- speaking video
- listening video
- thinking video

### C. Browser-rendered 3D avatar
- GLB
- Three.js/WebGL
- facial expressions
- visemes
- gestures
- idle animations

A strong open-source candidate is **TalkingHead**, which renders full-body GLB avatars in the browser using Three.js/WebGL and supports lip-sync, facial expressions and animations.

Another useful approach is browser-side MediaPipe Face Landmarker, which can provide face landmarks/blendshape-related information entirely in the web application.

For lightweight 2D avatars, **Voqalize Avatar** is particularly interesting: it provides Canvas/SVG talking-head avatars, browser-side rendering and lip-sync without requiring a video stream; its repository is MIT licensed.

There are also browser projects supporting VRM/Live2D with client-side animation, which could be evaluated as an alternative implementation path.

### Recommended architecture

For the commercial product I would support **three avatar rendering modes behind one abstraction**:

`Avatar Engine`
- `StaticImageAvatar`
- `LoopVideoAvatar`
- `WebGL3DAvatar`

This prevents the chat application from becoming coupled to a particular avatar technology.

---

# 14. Avatar Administration

Admin GUI:

- upload avatar
- preview avatar
- assign avatar to personality
- configure idle state
- configure listening state
- configure thinking state
- configure speaking state
- configure animations
- configure expressions
- configure visemes
- configure background
- configure avatar size/crop
- configure fallback avatar

Avatar assets should have:
- version
- resolution
- format
- file size
- preview
- status
- CDN/cache information

---

# 15. TTS / STT

## 15.1 STT

Speech-to-text should preferably execute on the client when technically feasible.

Support:
- microphone input
- push-to-talk
- continuous dictation
- voice activity detection
- language selection
- transcription preview
- correction before sending
- interruption/barge-in

## 15.2 TTS

Architecture should support:

### Client-side TTS
For low-cost/basic interactions.

### Server/provider TTS
For premium-quality voices.

The TTS layer should return timing information whenever possible:

- word timestamps
- phonemes
- visemes

This allows the browser avatar engine to synchronize animation without generating video server-side.

---

# 16. Subscription & Monetization

## 16.1 Credit model

Commercial consumption should be based primarily on **responses**, not raw tokens.

Internally maintain token accounting for cost control.

Example:

User sees:

`342 responses remaining`

Backend tracks:

`input tokens`
`output tokens`
`provider cost`
`model usage`

This abstracts provider/model differences from the customer.

## 16.2 Subscription plans

Plans should support configurable:

- monthly price
- annual price
- daily response limit
- weekly response limit
- monthly response limit
- personality categories available
- personality types available
- premium model access
- voice access
- avatar access
- memory capacity
- history capacity
- priority
- maximum conversation length

## 16.3 Packages

One-time credit packages:

- 50 responses
- 100 responses
- 500 responses
- custom packages

Packages should have configurable expiration policies.

## 16.4 Subscription management

Admin can configure:

- plan
- price
- quotas
- included responses
- available personality categories
- discounts
- coupons
- promotional pricing
- referral rewards
- free trials
- special offers
- introductory pricing
- renewal behavior

## 16.5 Entitlement engine

Recommended addition:

Do not hard-code subscription rules into the frontend.

Create a backend **Entitlement Engine**.

Example:

`canAccess(user, personality)`

`canSendMessage(user)`

`canUseVoice(user)`

`canUsePremiumModel(user)`

`canUseMemory(user)`

This makes monetization rules configurable without rewriting application logic.

---

# 17. Billing

Recommended dedicated billing subsystem:

- payment provider integration
- checkout
- subscriptions
- invoices
- refunds
- failed payments
- cancellations
- renewals
- webhook processing
- payment history

Initially payment providers can be mocked, but the domain model should already support production billing.

---

# 18. Referral / Coupon / Promotion System

Support:

- coupon codes
- percentage discount
- fixed discount
- free responses
- bonus responses
- referral codes
- referral rewards
- trial periods
- first-purchase offers
- limited-time offers
- personality-specific promotions

Admin should be able to define:

`promotion → target users → personality category → period → conditions → benefit`

---

# 19. User Usage & Statistics

User dashboard should show:

- responses used
- responses remaining
- subscription
- renewal date
- usage by personality
- conversations
- voice usage
- memory usage
- usage over time

Charts:
- daily
- weekly
- monthly

---

# 20. Admin Analytics

Create a dedicated analytics subsystem.

## 20.1 System-level metrics

Track:

- total users
- active users
- new users
- returning users
- DAU
- WAU
- MAU
- retention
- sessions
- conversations
- messages
- responses
- average response time
- p50 latency
- p95 latency
- p99 latency
- error rate
- timeout rate
- provider failure rate

## 20.2 LLM metrics

- input tokens
- output tokens
- total tokens
- tokens/request
- tokens/user
- tokens/personality
- tokens/model
- provider cost
- estimated cost/user
- cost/personality
- cost/day
- cost/month

## 20.3 Infrastructure metrics

- CPU utilization
- RAM utilization
- GPU utilization
- VRAM utilization
- network traffic
- request concurrency
- queue depth
- worker utilization
- provider saturation
- database latency
- vector DB latency
- cache hit rate

## 20.4 Time analysis

Display usage by:

- hour
- day
- weekday
- week
- month

This allows identification of peak load periods.

## 20.5 Personality analytics

For each personality:

- conversations
- unique users
- responses
- average session length
- average messages/session
- retention
- popularity
- voice usage
- memory creation
- average response latency
- error rate
- cost
- revenue attribution

## 20.6 Subscription analytics

Track:

- subscribers by plan
- free → paid conversion
- trial → paid conversion
- churn
- renewal rate
- ARPU
- revenue
- revenue/personality category
- credit purchases
- coupon usage
- referral conversions

## 20.7 Funnel analytics

Recommended addition:

`Registration → First chat → First response → Second session → Trial → Paid → Renewal`

This is essential for understanding commercial performance.

---

# 21. Observability

Implement distributed tracing from the beginning.

Each request should carry:

- request ID
- correlation ID
- user ID
- conversation ID
- personality ID
- model ID
- provider ID

Trace:

`Frontend`
→ `API`
→ `Orchestrator`
→ `Memory`
→ `RAG/CAG`
→ `LLM`
→ `Guardrails`
→ `Ground-check`
→ `TTS`
→ `Frontend`

Measure latency of every stage separately.

This is substantially more useful than having only "total response time".

---

# 22. Audit System

All administrative and security-sensitive operations should be audited.

Log:

- user creation/deletion
- subscription changes
- personality changes
- prompt changes
- RAG changes
- CAG changes
- guardrail changes
- provider changes
- model changes
- memory access by administrator
- memory deletion
- user data export
- billing changes

Each audit record:

- actor
- action
- target
- timestamp
- before/after state where appropriate
- IP/device metadata where appropriate
- correlation ID

---

# 23. Security

Implement from the beginning:

- RBAC
- API authentication
- authorization on every protected resource
- tenant isolation
- rate limiting
- abuse prevention
- API key encryption
- secrets management
- secure session handling
- audit logs
- input validation
- output validation
- prompt injection protection
- administrator 2FA
- data deletion workflows

---

# 24. Data & Privacy Controls

Recommended addition.

The system should distinguish:

- account data
- chat data
- memory data
- analytics data
- billing data
- system logs

User controls:

- export data
- delete account
- delete chats
- delete memories
- delete personality-specific memories
- disable memory
- clear conversation history

Retention policies should be configurable by data type.

---

# 25. Admin Dashboard Structure

The Admin GUI should be organized approximately as:

### 25.1 Dashboard
- system health
- active users
- active conversations
- latency
- errors
- token usage
- costs
- infrastructure
- alerts

### 25.2 Users
- user list
- user details
- subscription
- usage
- conversations
- memories
- statistics
- account actions

### 25.3 Personalities
- list
- create
- edit
- clone
- categories
- types
- avatars
- prompts
- RAG
- CAG
- guardrails
- ground-check
- models
- TTS/STT
- version history

### 25.4 Knowledge
- RAG
- CAG
- documents
- indexes
- retrieval testing

### 25.5 AI Providers
- providers
- models
- health
- routing
- costs

### 25.6 Memory
- users
- personalities
- memories
- search
- inspection
- export

### 25.7 Conversations
- users
- personalities
- search
- inspection
- export

### 25.8 Subscriptions
- plans
- packages
- prices
- quotas
- entitlements

### 25.9 Promotions
- coupons
- referrals
- trials
- special offers

### 25.10 Analytics
- users
- usage
- revenue
- AI costs
- latency
- infrastructure
- personalities

### 25.11 System
- infrastructure
- queues
- workers
- databases
- caches
- health
- logs
- audit

---

# 26. Recommended Additional Features

## 26.1 Feature flags

Implement feature flags from the beginning.

Allows enabling features for:
- specific users
- plans
- personalities
- percentages of users
- beta testers

## 26.2 A/B testing

Useful for:
- chat UI
- onboarding
- personality presentation
- pricing
- subscription offers
- avatar types
- prompts

## 26.3 Personality evaluation

Admin should be able to run predefined test conversations against a personality.

Compare:
- model versions
- prompt versions
- guardrail versions
- RAG versions

Metrics:
- factuality
- consistency
- latency
- cost
- response quality
- ground-check failures

This becomes an internal **Personality Evaluation Framework**.

## 26.4 Prompt/configuration playground

Admin can test:

`Personality + Model + RAG + CAG + Guardrails + Ground-check`

against an input before publishing changes.

## 26.5 Cost simulation

Before activating a personality/model combination:

- estimated cost/response
- estimated cost/user
- projected monthly cost
- maximum sustainable concurrency

This is particularly important when premium personalities use expensive models.

---

# 27. Suggested Core Domain Model

At the conceptual level, the system should revolve around:

```text
User
 ├── Subscription
 ├── Credits
 ├── Conversations
 │    └── Messages
 ├── Memories
 ├── Usage
 └── Preferences

Personality
 ├── PersonalityTypes[]
 ├── CommercialCategory
 ├── Avatar
 ├── Prompt
 ├── RAG[]
 ├── CAG[]
 ├── Guardrails
 ├── GroundCheck
 ├── LLMConfiguration
 ├── TTSConfiguration
 ├── STTConfiguration
 └── MemoryConfiguration

Subscription
 ├── Entitlements
 ├── Quotas
 ├── Pricing
 ├── Discounts
 └── Promotions

LLMProvider
 └── Models[]

Conversation
 ├── User
 ├── Personality
 ├── Messages[]
 └── MemoryReferences[]

Memory
 ├── User
 ├── Personality
 ├── Type
 ├── Content
 ├── TemporalMetadata
 ├── Importance
 ├── Confidence
 └── Source

RAGSource
 └── Personality[]

CAGSource
 └── Personality[]
```

---

# 28. Implementation Priority

## Phase 1 — Platform Foundation
1. Multi-user architecture
2. Frontend/backend separation
3. Authentication
4. RBAC
5. User management
6. Database/domain model
7. Admin framework
8. Audit logging

## Phase 2 — AI Core
9. Personality management
10. Personality types
11. Commercial categories
12. LLM providers
13. Model routing
14. RAG
15. CAG
16. Guardrails
17. Ground-check

## Phase 3 — Chat & Memory
18. Chat engine
19. Conversation history
20. Short-term memory
21. Long-term memory
22. Temporal memory
23. Automatic memory retrieval
24. Memory consolidation
25. Cross-personality memory references
26. User memory management
27. Admin memory browser

## Phase 4 — Avatar & Voice
28. Client-side avatar engine
29. Avatar administration
30. Client STT
31. TTS
32. Lip-sync
33. Voice interruption/barge-in
34. Multiple avatar implementations

## Phase 5 — Monetization
35. Credit/response system
36. Subscription plans
37. Entitlement engine
38. Packages
39. Billing
40. Coupons
41. Referrals
42. Trials
43. Promotions

## Phase 6 — Analytics & Operations
44. User analytics
45. Personality analytics
46. LLM analytics
47. Cost analytics
48. Infrastructure monitoring
49. Latency monitoring
50. Distributed tracing
51. Funnel analytics
52. Reports
53. Alerts

## Phase 7 — Commercial Polish
54. Premium chat UI
55. Advanced personality discovery
56. Onboarding
57. Recommendations
58. A/B testing
59. Personality evaluation
60. Admin playground
61. Feature flags
62. Performance optimization
63. Final UX/UI polish

---

# 29. Critical Architectural Principle

The most important structural decision is to **avoid coupling the application to any specific LLM, avatar engine, TTS provider or payment provider**.

The architecture should therefore expose internal abstractions such as:

```text
LLMProvider
MemoryProvider
RAGProvider
CAGProvider
TTSProvider
STTProvider
AvatarEngine
BillingProvider
```

This allows the platform to change:

- OpenAI → another LLM
- cloud → local model
- one TTS → another TTS
- one avatar technology → another
- Stripe → another payment provider

without rewriting the core application.

The same principle should apply to memory: **conversation history, short-term memory, long-term memory, semantic retrieval and temporal retrieval should be separate capabilities behind a common Memory Service.**