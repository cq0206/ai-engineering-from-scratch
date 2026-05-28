# 缓存、速率限制与成本优化

> 大多数 AI 初创公司不是死于模型太差，而是死于糟糕的单位经济（unit economics）。一次 GPT-4o 调用只花几分之一美分。但如果一万名用户每天各发起十次调用，仅输入 token 成本就要 $250——而这还发生在你一分钱收入都没收到之前。真正能活下来的公司，是把每一次 API 调用都当成一笔金融交易，而不是一次函数调用来对待。

**类型：** 构建
**语言：** Python
**先修要求：** 第 11 阶段 第 09 课（Function Calling）
**时间：** ~45 分钟
**相关内容：** 第 11 阶段 · 15（提示缓存 Prompt Caching）——本课介绍应用层缓存（application-layer caching），包括语义缓存（semantic cache）、精确哈希缓存（exact hash cache）和模型路由（model routing）。第 15 课介绍提供商层提示缓存（provider-layer prompt caching），包括 Anthropic `cache_control`、OpenAI 自动缓存和 Gemini `CachedContent`。两者结合可实现 50-95% 的成本下降。

## 学习目标

- 实现语义缓存，让重复或相似的查询直接从缓存返回，而不是再次发起 API 调用
- 计算跨提供商的单次请求成本，并实现具备 token 感知能力的速率限制与预算告警
- 构建一个成本优化层，结合提示压缩、模型路由（昂贵模型 vs 廉价模型）和响应缓存
- 针对不同类型的查询，设计结合精确匹配、语义相似度和前缀缓存的分层缓存策略

## 问题所在

你做了一个 RAG 聊天机器人。效果很棒。用户也很喜欢。

然后，账单来了。

GPT-5 的价格是每百万输入 tokens $5、每百万输出 $15。Claude Opus 4.7 是输入 $15 / 输出 $75。Gemini 3 Pro 是输入 $1.25 / 输出 $5。GPT-5-mini 是 $0.25/$2。下面的价格仅作示意；请始终以提供商当前的定价页面为准。

下面这笔账，足以杀死许多初创公司：

- 每日活跃用户 10,000
- 每位用户每天 10 次查询
- 每次查询 1,000 个输入 tokens（system prompt + context + user message）
- 每次响应 500 个输出 tokens

**每日输入成本：** 10,000 x 10 x 1,000 / 1,000,000 x $2.50 = **$250/天**
**每日输出成本：** 10,000 x 10 x 500 / 1,000,000 x $10.00 = **$500/天**
**月总成本：** **$22,500/月**

这还只是 LLM 本身的成本。再加上嵌入（embeddings）、向量数据库托管和基础设施，一个聊天机器人每月花到 $30,000 并不夸张。

更残酷的是：其中 40-60% 的查询都近似重复。用户只是换了几种说法问同一个问题。你的系统提示词——每次请求都完全相同——却会被一遍又一遍计费。RAG 检索出的上下文文档，在不同用户询问同一主题时也会不断重复。

你正在为冗余计算支付全价。

## 核心概念

### 一次 LLM 调用的成本结构

每一次 API 调用都包含五个成本组成部分。

```mermaid
graph LR
    A[用户查询] --> B[系统提示词<br/>500-2000 tokens]
    A --> C[检索到的上下文<br/>500-4000 tokens]
    A --> D[用户消息<br/>50-500 tokens]
    B --> E[输入成本<br/>$2.50/1M tokens]
    C --> E
    D --> E
    E --> F[模型处理]
    F --> G[输出成本<br/>$10.00/1M tokens]
```

系统提示词是那个悄无声息的杀手。一个 1,500-token 的系统提示词，如果每次请求都发送，仅这一段前缀在每百万次请求上就要花 $3.75。若每天 100K 次请求，那就是 $375/天——$11,250/月——而这些文本压根从未变化。

### 提供商缓存：内建折扣

到 2026 年，三大主流提供商都提供了提供商侧提示缓存，但具体机制各不相同。深入内容见第 11 阶段 · 15。

| 提供商 | 机制 | 折扣 | 最低要求 | 缓存时长 |
|----------|-----------|----------|---------|----------------|
| Anthropic | 显式 `cache_control` 标记 | 缓存命中时 90% 折扣（写入时额外支付 25%） | 1,024 tokens（Sonnet/Opus），2,048（Haiku） | 默认 5 分钟；扩展到 1 小时（写入溢价变为 2x） |
| OpenAI | 自动前缀匹配 | 缓存命中时 50% 折扣 | 1,024 tokens | 尽力而为，最长约 1 小时 |
| Google Gemini | 显式 `CachedContent` API | 约 75% 降幅（外加存储费用） | 4,096（Flash）/ 32,768（Pro） | 用户可配置 TTL |

**Anthropic 的做法**是显式控制。你需要用 `cache_control: {"type": "ephemeral"}` 标记提示词中的某些部分。第一次请求要支付 25% 的写入溢价。之后只要前缀相同，就能享受 90% 折扣。一个原本成本为 $0.005 的 2,000-token 系统提示词，在缓存命中时只要 $0.000625。按 100K 次请求计算，这每天能省下 $437.50。

**OpenAI 的做法**是自动的。任何与先前请求前缀匹配的提示词，都会自动获得 50% 折扣。不需要任何标记。权衡在于：折扣更低、控制更少，但实现成本为零。

### 语义缓存：你的自定义层

提供商缓存只适用于完全相同的前缀。语义缓存处理的是更难的一类情况：字面不同，但含义相同的查询。

“退货政策是什么？”和“我该如何退货？”是不同的字符串，但意图相同。语义缓存会为两者生成嵌入，计算余弦相似度，并在相似度超过阈值时返回缓存响应（通常是 0.92-0.95）。

```mermaid
flowchart TD
    A[用户查询] --> B[嵌入查询]
    B --> C{缓存中有<br/>相似查询吗？}
    C -->|sim > 0.95| D[返回缓存响应]
    C -->|sim < 0.95| E[调用 LLM API]
    E --> F[连同嵌入缓存响应]
    F --> G[返回响应]
    D --> G
```

嵌入的成本几乎可以忽略不计。OpenAI 的 `text-embedding-3-small` 每百万 tokens 只需 $0.02。与一次完整 LLM 调用相比，做一次缓存检查几乎不花钱。

### 精确缓存：哈希并匹配

对于确定性调用（temperature=0、模型相同、提示词相同），精确缓存更简单也更快。对完整提示词做哈希，查缓存，命中就直接返回。

它非常适合以下场景：
- 系统提示词 + 固定上下文 + 完全相同的用户查询
- 使用完全相同工具定义的 function calling
- 同一文档会被重复处理多次的批处理任务

### 速率限制：保护你的预算

速率限制不只是为了公平，更是为了生存。

**令牌桶算法（token bucket algorithm）：** 每个用户都有一个包含 N 个 token 的桶，并按每秒 R 个 token 的速度回填。每个请求都会消耗桶中的 token。若桶为空，请求就会被拒绝。这种方式既允许突发流量（一次性用完整个桶），又能约束长期平均速率。

**按用户配额：** 为不同用户层级设置每日/月度 token 限额。

| 级别 | 每日 Token 上限 | 每分钟最大请求数 | 可用模型 |
|------|------------------|------------------|-------------|
| 免费版 | 50,000 | 10 | 仅 GPT-4o-mini |
| 专业版 | 500,000 | 60 | GPT-4o、Claude Sonnet |
| 企业版 | 5,000,000 | 300 | 所有模型 |

### 模型路由：为不同任务选择合适模型

不是每个查询都需要 GPT-4o。

“商店几点关门？”并不需要一个输出价格 $10/M 的模型。输出价格 $0.60/M 的 GPT-4o-mini 就完全够用。输出价格 $1.25/M 的 Claude Haiku 也能胜任。一个简单的分类器，就能把便宜查询路由到便宜模型，把复杂查询路由到昂贵模型。

```mermaid
flowchart TD
    A[用户查询] --> B[复杂度分类器]
    B -->|简单：查找、FAQ| C[GPT-4o-mini<br/>$0.15/$0.60 per 1M]
    B -->|中等：分析、摘要| D[Claude Sonnet<br/>$3.00/$15.00 per 1M]
    B -->|复杂：推理、代码| E[GPT-4o / Claude Opus<br/>$2.50/$10.00+]
```

一个调优良好的路由器，单靠模型成本就能节省 40-70%。

### 成本追踪：知道钱花到哪里去了

你无法优化自己无法衡量的东西。为每一次 API 调用记录以下信息：

- 时间戳
- 模型名称
- 输入 tokens
- 输出 tokens
- 延迟（ms）
- 计算后的成本（$）
- 用户 ID
- 缓存命中/未命中
- 请求类别

这些数据会告诉你：哪些功能最贵、哪些用户是重度消耗者、以及缓存在哪些地方最有价值。

### 批处理：批量折扣

OpenAI 的 Batch API 以异步方式处理请求，并提供 50% 折扣。你可以一次提交最多 50,000 个请求，结果会在 24 小时内返回。

适合使用批处理的场景：
- 夜间文档处理
- 大批量分类
- 评估运行
- 数据增强流水线

不适合：面向用户的实时查询（因为延迟很重要）。

### 预算告警与电路断路器

电路断路器（circuit breaker）会在你触达某个限制时停止继续花钱。没有它的话，一个 bug 或一次滥用可能在几个小时内烧光你整月的预算。

设置三个阈值：
1. **警告**（预算的 70%）：发送告警
2. **限流**（预算的 85%）：只切换到更便宜的模型
3. **停止**（预算的 95%）：拒绝新请求，只返回缓存响应

### 优化栈

按顺序应用这些技术。每一层都会叠加前一层的效果。

| 层级 | 技术 | 典型节省比例 | 实现工作量 |
|-------|-----------|----------------|----------------------|
| 1 | 提供商提示缓存 | 30-50% | 低（添加缓存标记） |
| 2 | 精确缓存 | 10-20% | 低（hash + dict） |
| 3 | 语义缓存 | 15-30% | 中（embeddings + similarity） |
| 4 | 模型路由 | 40-70% | 中（classifier） |
| 5 | 速率限制 | 预算保护 | 低（token bucket） |
| 6 | 提示压缩 | 10-30% | 中（重写提示词） |
| 7 | 批处理 | 适用场景 50% | 低（batch API） |

一个应用了第 1-5 层的 RAG 应用，通常能把成本从 $22,500/月 降到 $4,000-6,000/月。这就是“烧光 runway”和“做成一门生意”之间的差别。

### 真实节省：优化前 vs 优化后

下面是一个服务 10,000 DAU 的 RAG 聊天机器人的真实拆解。

| 指标 | 优化前 | 优化后 | 节省 |
|--------|--------------------|--------------------|---------|
| 每月 LLM 成本 | $22,500 | $5,200 | 77% |
| 每次查询平均成本 | $0.0075 | $0.0017 | 77% |
| 缓存命中率 | 0% | 52% | -- |
| 路由到 mini 的查询占比 | 0% | 65% | -- |
| P95 延迟 | 2,800ms | 900ms（缓存命中：50ms） | 68% |
| 每月嵌入成本 | $0 | $180 | （新增成本） |
| 每月总成本 | $22,500 | $5,380 | 76% |

语义缓存带来的嵌入成本（$180/月），只要缓存命中开始出现，在第一个小时内就能回本。

## 动手构建

### 步骤 1：成本计算器

构建一个 token 成本计算器，了解主流模型的当前定价。

```python
import hashlib
import time
import json
import math
from dataclasses import dataclass, field


MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075},
    "gpt-4.1": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cached_input": 0.10},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cached_input": 0.025},
    "o3": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
    "o3-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.55},
    "o4-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.275},
    "claude-opus-4": {"input": 15.00, "output": 75.00, "cached_input": 1.50},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-haiku-3.5": {"input": 0.80, "output": 4.00, "cached_input": 0.08},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00, "cached_input": 0.3125},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60, "cached_input": 0.0375},
}


def calculate_cost(model, input_tokens, output_tokens, cached_input_tokens=0):
    if model not in MODEL_PRICING:
        return {"error": f"Unknown model: {model}"}
    pricing = MODEL_PRICING[model]
    non_cached = input_tokens - cached_input_tokens
    input_cost = (non_cached / 1_000_000) * pricing["input"]
    cached_cost = (cached_input_tokens / 1_000_000) * pricing["cached_input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total = input_cost + cached_cost + output_cost
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "input_cost": round(input_cost, 6),
        "cached_input_cost": round(cached_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total, 6),
    }
```

### 步骤 2：精确缓存

对完整提示词做哈希，并为完全相同的请求返回缓存响应。

```python
class ExactCache:
    def __init__(self, max_size=1000, ttl_seconds=3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _hash(self, model, messages, temperature):
        key_data = json.dumps({"model": model, "messages": messages, "temperature": temperature}, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, model, messages, temperature=0.0):
        if temperature > 0:
            self.misses += 1
            return None
        key = self._hash(model, messages, temperature)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                self.hits += 1
                entry["access_count"] += 1
                return entry["response"]
            del self.cache[key]
        self.misses += 1
        return None

    def put(self, model, messages, temperature, response):
        if temperature > 0:
            return
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache, key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        key = self._hash(model, messages, temperature)
        self.cache[key] = {
            "response": response,
            "timestamp": time.time(),
            "access_count": 1,
        }

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0,
            "cache_size": len(self.cache),
        }
```

### 步骤 3：语义缓存

为查询生成嵌入，并在相似度超过阈值时返回缓存响应。

```python
def simple_embed(text):
    words = text.lower().split()
    vocab = {}
    for w in words:
        vocab[w] = vocab.get(w, 0) + 1
    norm = math.sqrt(sum(v * v for v in vocab.values()))
    if norm == 0:
        return {}
    return {k: v / norm for k, v in vocab.items()}


def cosine_similarity(a, b):
    if not a or not b:
        return 0.0
    all_keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in all_keys)
    return dot


class SemanticCache:
    def __init__(self, similarity_threshold=0.85, max_size=500, ttl_seconds=3600):
        self.entries = []
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def get(self, query):
        query_embedding = simple_embed(query)
        now = time.time()
        best_match = None
        best_sim = 0.0
        for entry in self.entries:
            if now - entry["timestamp"] > self.ttl:
                continue
            sim = cosine_similarity(query_embedding, entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_match = entry
        if best_match and best_sim >= self.threshold:
            self.hits += 1
            best_match["access_count"] += 1
            return {"response": best_match["response"], "similarity": round(best_sim, 4), "original_query": best_match["query"]}
        self.misses += 1
        return None

    def put(self, query, response):
        if len(self.entries) >= self.max_size:
            self.entries.sort(key=lambda e: e["timestamp"])
            self.entries.pop(0)
        self.entries.append({
            "query": query,
            "embedding": simple_embed(query),
            "response": response,
            "timestamp": time.time(),
            "access_count": 1,
        })

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0,
            "cache_size": len(self.entries),
        }
```

### 步骤 4：速率限制器

使用带有按用户配额的令牌桶速率限制器。

```python
class TokenBucketRateLimiter:
    def __init__(self):
        self.buckets = {}
        self.tiers = {
            "free": {"capacity": 50_000, "refill_rate": 500, "max_requests_per_min": 10},
            "pro": {"capacity": 500_000, "refill_rate": 5_000, "max_requests_per_min": 60},
            "enterprise": {"capacity": 5_000_000, "refill_rate": 50_000, "max_requests_per_min": 300},
        }

    def _get_bucket(self, user_id, tier="free"):
        if user_id not in self.buckets:
            tier_config = self.tiers.get(tier, self.tiers["free"])
            self.buckets[user_id] = {
                "tokens": tier_config["capacity"],
                "capacity": tier_config["capacity"],
                "refill_rate": tier_config["refill_rate"],
                "last_refill": time.time(),
                "request_timestamps": [],
                "max_rpm": tier_config["max_requests_per_min"],
                "tier": tier,
                "total_tokens_used": 0,
            }
        return self.buckets[user_id]

    def _refill(self, bucket):
        now = time.time()
        elapsed = now - bucket["last_refill"]
        refill = int(elapsed * bucket["refill_rate"])
        if refill > 0:
            bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + refill)
            bucket["last_refill"] = now

    def check(self, user_id, tokens_needed, tier="free"):
        bucket = self._get_bucket(user_id, tier)
        self._refill(bucket)
        now = time.time()
        bucket["request_timestamps"] = [t for t in bucket["request_timestamps"] if now - t < 60]
        if len(bucket["request_timestamps"]) >= bucket["max_rpm"]:
            return {"allowed": False, "reason": "rate_limit", "retry_after_seconds": 60 - (now - bucket["request_timestamps"][0])}
        if bucket["tokens"] < tokens_needed:
            deficit = tokens_needed - bucket["tokens"]
            wait = deficit / bucket["refill_rate"]
            return {"allowed": False, "reason": "token_limit", "tokens_available": bucket["tokens"], "retry_after_seconds": round(wait, 1)}
        return {"allowed": True, "tokens_available": bucket["tokens"]}

    def consume(self, user_id, tokens_used, tier="free"):
        bucket = self._get_bucket(user_id, tier)
        bucket["tokens"] -= tokens_used
        bucket["request_timestamps"].append(time.time())
        bucket["total_tokens_used"] += tokens_used

    def get_usage(self, user_id):
        if user_id not in self.buckets:
            return {"error": "User not found"}
        b = self.buckets[user_id]
        return {
            "user_id": user_id,
            "tier": b["tier"],
            "tokens_remaining": b["tokens"],
            "capacity": b["capacity"],
            "total_tokens_used": b["total_tokens_used"],
            "utilization": round(b["total_tokens_used"] / b["capacity"], 4) if b["capacity"] else 0,
        }
```

### 步骤 5：成本追踪器

记录每一次调用，并计算滚动总量。

```python
class CostTracker:
    def __init__(self, monthly_budget=1000.0):
        self.logs = []
        self.monthly_budget = monthly_budget
        self.alerts = []

    def log_call(self, model, input_tokens, output_tokens, cached_input_tokens=0, latency_ms=0, user_id="anonymous", cache_status="miss"):
        cost = calculate_cost(model, input_tokens, output_tokens, cached_input_tokens)
        entry = {
            "timestamp": time.time(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "latency_ms": latency_ms,
            "cost": cost["total_cost"],
            "user_id": user_id,
            "cache_status": cache_status,
        }
        self.logs.append(entry)
        self._check_budget()
        return entry

    def _check_budget(self):
        total = self.total_cost()
        pct = total / self.monthly_budget if self.monthly_budget > 0 else 0
        if pct >= 0.95 and not any(a["level"] == "stop" for a in self.alerts):
            self.alerts.append({"level": "stop", "message": f"Budget 95% consumed: ${total:.2f}/${self.monthly_budget:.2f}", "timestamp": time.time()})
        elif pct >= 0.85 and not any(a["level"] == "throttle" for a in self.alerts):
            self.alerts.append({"level": "throttle", "message": f"Budget 85% consumed: ${total:.2f}/${self.monthly_budget:.2f}", "timestamp": time.time()})
        elif pct >= 0.70 and not any(a["level"] == "warning" for a in self.alerts):
            self.alerts.append({"level": "warning", "message": f"Budget 70% consumed: ${total:.2f}/${self.monthly_budget:.2f}", "timestamp": time.time()})

    def total_cost(self):
        return round(sum(e["cost"] for e in self.logs), 6)

    def cost_by_model(self):
        by_model = {}
        for e in self.logs:
            m = e["model"]
            if m not in by_model:
                by_model[m] = {"calls": 0, "cost": 0, "input_tokens": 0, "output_tokens": 0}
            by_model[m]["calls"] += 1
            by_model[m]["cost"] = round(by_model[m]["cost"] + e["cost"], 6)
            by_model[m]["input_tokens"] += e["input_tokens"]
            by_model[m]["output_tokens"] += e["output_tokens"]
        return by_model

    def cache_savings(self):
        cache_hits = [e for e in self.logs if e["cache_status"] == "hit"]
        if not cache_hits:
            return {"saved": 0, "cache_hits": 0}
        saved = 0
        for e in cache_hits:
            full_cost = calculate_cost(e["model"], e["input_tokens"], e["output_tokens"])
            saved += full_cost["total_cost"]
        return {"saved": round(saved, 4), "cache_hits": len(cache_hits)}

    def summary(self):
        if not self.logs:
            return {"total_calls": 0, "total_cost": 0}
        total_latency = sum(e["latency_ms"] for e in self.logs)
        cache_hits = sum(1 for e in self.logs if e["cache_status"] == "hit")
        return {
            "total_calls": len(self.logs),
            "total_cost": self.total_cost(),
            "avg_cost_per_call": round(self.total_cost() / len(self.logs), 6),
            "avg_latency_ms": round(total_latency / len(self.logs), 1),
            "cache_hit_rate": round(cache_hits / len(self.logs), 4),
            "cost_by_model": self.cost_by_model(),
            "cache_savings": self.cache_savings(),
            "budget_remaining": round(self.monthly_budget - self.total_cost(), 2),
            "budget_utilization": round(self.total_cost() / self.monthly_budget, 4) if self.monthly_budget > 0 else 0,
            "alerts": self.alerts,
        }
```

### 步骤 6：模型路由器

将查询路由到能胜任任务的最便宜模型。

```python
SIMPLE_KEYWORDS = ["what time", "hours", "address", "phone", "price", "return policy", "hello", "hi", "thanks", "yes", "no"]
COMPLEX_KEYWORDS = ["analyze", "compare", "explain why", "write code", "debug", "architect", "design", "trade-off", "evaluate"]


def classify_complexity(query):
    q = query.lower()
    if len(q.split()) <= 5 or any(kw in q for kw in SIMPLE_KEYWORDS):
        return "simple"
    if any(kw in q for kw in COMPLEX_KEYWORDS):
        return "complex"
    return "medium"


def route_model(query, tier="pro"):
    complexity = classify_complexity(query)
    routing_table = {
        "simple": {"free": "gpt-4.1-nano", "pro": "gpt-4o-mini", "enterprise": "gpt-4o-mini"},
        "medium": {"free": "gpt-4o-mini", "pro": "claude-sonnet-4", "enterprise": "claude-sonnet-4"},
        "complex": {"free": "gpt-4o-mini", "pro": "gpt-4o", "enterprise": "claude-opus-4"},
    }
    model = routing_table[complexity].get(tier, "gpt-4o-mini")
    return {"query": query, "complexity": complexity, "model": model, "tier": tier}
```

### 步骤 7：运行演示

```python
def simulate_llm_call(model, query):
    input_tokens = len(query.split()) * 4 + 500
    output_tokens = 150 + (len(query.split()) * 2)
    latency = 200 + (output_tokens * 2)
    return {
        "model": model,
        "response": f"[Simulated {model} response to: {query[:50]}...]",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency,
    }


def run_demo():
    print("=" * 60)
    print("  Caching, Rate Limiting & Cost Optimization Demo")
    print("=" * 60)

    print("\n--- Model Pricing ---")
    for model, pricing in list(MODEL_PRICING.items())[:6]:
        cost_1k = calculate_cost(model, 1000, 500)
        print(f"  {model}: ${cost_1k['total_cost']:.6f} per 1K in + 500 out")

    print("\n--- Cost Comparison: 100K Requests ---")
    for model in ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4", "claude-haiku-3.5"]:
        cost = calculate_cost(model, 1000 * 100_000, 500 * 100_000)
        print(f"  {model}: ${cost['total_cost']:.2f}")

    print("\n--- Anthropic Cache Savings ---")
    no_cache = calculate_cost("claude-sonnet-4", 2000, 500, 0)
    with_cache = calculate_cost("claude-sonnet-4", 2000, 500, 1500)
    saving = no_cache["total_cost"] - with_cache["total_cost"]
    print(f"  Without cache: ${no_cache['total_cost']:.6f}")
    print(f"  With 1500 cached tokens: ${with_cache['total_cost']:.6f}")
    print(f"  Savings per call: ${saving:.6f} ({saving/no_cache['total_cost']*100:.1f}%)")

    exact_cache = ExactCache(max_size=100, ttl_seconds=300)
    semantic_cache = SemanticCache(similarity_threshold=0.75, max_size=100)
    rate_limiter = TokenBucketRateLimiter()
    tracker = CostTracker(monthly_budget=100.0)

    print("\n--- Exact Cache ---")
    messages_1 = [{"role": "user", "content": "What is the return policy?"}]
    result = exact_cache.get("gpt-4o-mini", messages_1, 0.0)
    print(f"  First lookup: {'HIT' if result else 'MISS'}")
    exact_cache.put("gpt-4o-mini", messages_1, 0.0, "You can return items within 30 days.")
    result = exact_cache.get("gpt-4o-mini", messages_1, 0.0)
    print(f"  Second lookup: {'HIT' if result else 'MISS'} -> {result}")
    result = exact_cache.get("gpt-4o-mini", messages_1, 0.7)
    print(f"  With temp=0.7: {'HIT' if result else 'MISS (non-deterministic, skip cache)'}")
    print(f"  Stats: {exact_cache.stats()}")

    print("\n--- Semantic Cache ---")
    test_queries = [
        ("What is the return policy?", "Items can be returned within 30 days with receipt."),
        ("How do I return an item?", None),
        ("What are your store hours?", "We are open 9am-9pm Monday through Saturday."),
        ("When does the store open?", None),
        ("Tell me about quantum computing", "Quantum computers use qubits..."),
        ("Explain quantum mechanics", None),
    ]
    for query, response in test_queries:
        cached = semantic_cache.get(query)
        if cached:
            print(f"  '{query[:40]}' -> CACHE HIT (sim={cached['similarity']}, original='{cached['original_query'][:40]}')")
        elif response:
            semantic_cache.put(query, response)
            print(f"  '{query[:40]}' -> MISS (stored)")
        else:
            print(f"  '{query[:40]}' -> MISS (no match)")
    print(f"  Stats: {semantic_cache.stats()}")

    print("\n--- Rate Limiting ---")
    for i in range(12):
        check = rate_limiter.check("user_1", 1000, "free")
        if check["allowed"]:
            rate_limiter.consume("user_1", 1000, "free")
        status = "OK" if check["allowed"] else f"BLOCKED ({check['reason']})"
        if i < 5 or not check["allowed"]:
            print(f"  Request {i+1}: {status}")
    print(f"  Usage: {rate_limiter.get_usage('user_1')}")

    print("\n--- Model Routing ---")
    routing_queries = [
        "What time do you close?",
        "Summarize this quarterly earnings report",
        "Analyze the trade-offs between microservices and monoliths",
        "Hello",
        "Write code for a binary search tree with deletion",
    ]
    for q in routing_queries:
        route = route_model(q, "pro")
        print(f"  '{q[:50]}' -> {route['model']} ({route['complexity']})")

    print("\n--- Full Pipeline: Before vs After Optimization ---")
    queries = [
        "What is the return policy?",
        "How do I return something?",
        "What are your hours?",
        "When do you open?",
        "Explain the difference between TCP and UDP",
        "Compare TCP vs UDP protocols",
        "Hello",
        "What is your phone number?",
        "Write a Python function to sort a list",
        "Analyze the pros and cons of serverless architecture",
    ]

    print("\n  [Before: no caching, single model (gpt-4o)]")
    tracker_before = CostTracker(monthly_budget=1000.0)
    for q in queries:
        result = simulate_llm_call("gpt-4o", q)
        tracker_before.log_call("gpt-4o", result["input_tokens"], result["output_tokens"], latency_ms=result["latency_ms"], cache_status="miss")
    before = tracker_before.summary()
    print(f"  Total cost: ${before['total_cost']:.6f}")
    print(f"  Avg cost/call: ${before['avg_cost_per_call']:.6f}")
    print(f"  Avg latency: {before['avg_latency_ms']}ms")

    print("\n  [After: caching + routing + rate limiting]")
    exact_c = ExactCache()
    semantic_c = SemanticCache(similarity_threshold=0.75)
    tracker_after = CostTracker(monthly_budget=1000.0)

    for q in queries:
        messages = [{"role": "user", "content": q}]
        cached = exact_c.get("gpt-4o", messages, 0.0)
        if cached:
            tracker_after.log_call("gpt-4o-mini", 0, 0, latency_ms=5, cache_status="hit")
            continue
        sem_cached = semantic_c.get(q)
        if sem_cached:
            tracker_after.log_call("gpt-4o-mini", 0, 0, latency_ms=15, cache_status="hit")
            continue
        route = route_model(q)
        result = simulate_llm_call(route["model"], q)
        tracker_after.log_call(route["model"], result["input_tokens"], result["output_tokens"], latency_ms=result["latency_ms"], cache_status="miss")
        exact_c.put(route["model"], messages, 0.0, result["response"])
        semantic_c.put(q, result["response"])

    after = tracker_after.summary()
    print(f"  Total cost: ${after['total_cost']:.6f}")
    print(f"  Avg cost/call: ${after['avg_cost_per_call']:.6f}")
    print(f"  Avg latency: {after['avg_latency_ms']}ms")
    print(f"  Cache hit rate: {after['cache_hit_rate']:.0%}")

    if before["total_cost"] > 0:
        savings_pct = (1 - after["total_cost"] / before["total_cost"]) * 100
        print(f"\n  SAVINGS: {savings_pct:.1f}% cost reduction")
        print(f"  Latency improvement: {(1 - after['avg_latency_ms'] / before['avg_latency_ms']) * 100:.1f}% faster")

    print("\n--- Budget Alerts Demo ---")
    alert_tracker = CostTracker(monthly_budget=0.01)
    for i in range(5):
        alert_tracker.log_call("gpt-4o", 5000, 2000, latency_ms=500)
    print(f"  Total spent: ${alert_tracker.total_cost():.6f} / ${alert_tracker.monthly_budget}")
    for alert in alert_tracker.alerts:
        print(f"  ALERT [{alert['level'].upper()}]: {alert['message']}")

    print("\n--- Cost Breakdown by Model ---")
    multi_tracker = CostTracker(monthly_budget=500.0)
    for _ in range(50):
        multi_tracker.log_call("gpt-4o-mini", 800, 200, latency_ms=150)
    for _ in range(30):
        multi_tracker.log_call("claude-sonnet-4", 1500, 500, latency_ms=400)
    for _ in range(10):
        multi_tracker.log_call("gpt-4o", 2000, 800, latency_ms=600)
    for _ in range(10):
        multi_tracker.log_call("claude-opus-4", 3000, 1000, latency_ms=1200)
    breakdown = multi_tracker.cost_by_model()
    for model, data in sorted(breakdown.items(), key=lambda x: x[1]["cost"], reverse=True):
        print(f"  {model}: {data['calls']} calls, ${data['cost']:.6f}, {data['input_tokens']:,} in / {data['output_tokens']:,} out")
    print(f"  Total: ${multi_tracker.total_cost():.6f}")

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
```

## 使用它

### Anthropic 提示缓存

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=1024,
#     system=[
#         {
#             "type": "text",
#             "text": "You are a helpful customer support agent for Acme Corp...",
#             "cache_control": {"type": "ephemeral"},
#         }
#     ],
#     messages=[{"role": "user", "content": "What is the return policy?"}],
# )
#
# print(f"Input tokens: {response.usage.input_tokens}")
# print(f"Cache creation tokens: {response.usage.cache_creation_input_tokens}")
# print(f"Cache read tokens: {response.usage.cache_read_input_tokens}")
```

第一次调用会把内容写入缓存（25% 溢价）。之后只要系统提示词前缀相同，每次调用都会从缓存中读取（90% 折扣）。缓存持续 5 分钟，并且每次命中都会重置计时器。

### OpenAI 自动缓存

```python
# from openai import OpenAI
#
# client = OpenAI()
#
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "system", "content": "You are a helpful customer support agent..."},
#         {"role": "user", "content": "What is the return policy?"},
#     ],
# )
#
# print(f"Prompt tokens: {response.usage.prompt_tokens}")
# print(f"Cached tokens: {response.usage.prompt_tokens_details.cached_tokens}")
# print(f"Completion tokens: {response.usage.completion_tokens}")
```

OpenAI 会自动缓存。任何长度达到 1,024+ tokens、且与最近请求前缀匹配的提示词，都会获得 50% 折扣。不需要修改任何代码——只要检查响应中的 `prompt_tokens_details.cached_tokens`，就能确认它是否在工作。

### OpenAI 批处理 API

```python
# import json
# from openai import OpenAI
#
# client = OpenAI()
#
# requests = []
# for i, query in enumerate(queries):
#     requests.append({
#         "custom_id": f"request-{i}",
#         "method": "POST",
#         "url": "/v1/chat/completions",
#         "body": {
#             "model": "gpt-4o-mini",
#             "messages": [{"role": "user", "content": query}],
#         },
#     })
#
# with open("batch_input.jsonl", "w") as f:
#     for r in requests:
#         f.write(json.dumps(r) + "\n")
#
# batch_file = client.files.create(file=open("batch_input.jsonl", "rb"), purpose="batch")
# batch = client.batches.create(input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h")
# print(f"Batch ID: {batch.id}, Status: {batch.status}")
```

Batch API 对所有 tokens 提供统一 50% 折扣。结果会在 24 小时内返回。它非常适合非实时工作负载：评估、数据标注、批量摘要。

### 基于 Redis 的生产级语义缓存

```python
# import redis
# import numpy as np
# from openai import OpenAI
#
# r = redis.Redis()
# client = OpenAI()
#
# def get_embedding(text):
#     response = client.embeddings.create(model="text-embedding-3-small", input=text)
#     return response.data[0].embedding
#
# def semantic_cache_lookup(query, threshold=0.95):
#     query_emb = np.array(get_embedding(query))
#     keys = r.keys("cache:emb:*")
#     best_sim, best_key = 0, None
#     for key in keys:
#         stored_emb = np.frombuffer(r.get(key), dtype=np.float32)
#         sim = np.dot(query_emb, stored_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(stored_emb))
#         if sim > best_sim:
#             best_sim, best_key = sim, key
#     if best_sim >= threshold and best_key:
#         response_key = best_key.decode().replace("cache:emb:", "cache:resp:")
#         return r.get(response_key).decode()
#     return None
```

在生产环境中，请把线性扫描替换为向量索引（Redis Vector Search、Pinecone 或 pgvector）。线性扫描只适用于少于 1,000 条记录的情况。再往上，就应该用 ANN（approximate nearest neighbor）把查询复杂度降到 O(log n)。

## 交付它

本课会产出 `outputs/prompt-cost-optimizer.md` —— 一个可复用的提示词，用于分析你的 LLM 应用并给出带有预计节省额的具体成本优化建议。

它还会产出 `outputs/skill-cost-patterns.md` —— 一个决策框架，帮助你为自己的用例选择合适的缓存策略、速率限制配置以及模型路由规则。

## 练习

1. **为语义缓存实现 LRU 淘汰。** 用“最近最少使用”替换当前“最早写入优先”的淘汰策略。跟踪每个条目的最后访问时间，并在缓存满时淘汰最后访问时间最早的条目。对 100 次查询比较两种策略的命中率。

2. **构建一个成本预测工具。** 给定一份 API 调用日志（即 CostTracker 日志），基于最近 7 天平均值预测月度成本。要考虑工作日/周末模式。如果预测的月度成本超出预算 20% 以上，就触发告警。

3. **实现分层语义缓存。** 使用两个相似度阈值：0.98 作为高置信命中（立即返回），0.90 作为中等置信命中（返回时附带免责声明：“基于一个相似的历史问题……”）。跟踪每次命中来自哪个层级，并衡量用户满意度差异。

4. **构建一个模型路由分类器。** 用基于嵌入的分类器替换基于关键词的分类器。先为 50 条已标注查询（simple/medium/complex）生成嵌入，再通过查找最近的已标注样本来分类新查询。用 20 条查询组成的测试集衡量分类准确率。

5. **实现一个带退化级别的电路断路器。** 当预算达到 70% 时记录警告。达到 85% 时，自动把所有路由切换到最便宜的模型（gpt-4o-mini）。达到 95% 时，只提供缓存响应并拒绝新查询。通过在 $1.00 预算下模拟 1,000 次请求来测试，并验证每个阈值都会正确触发。

## 关键术语

| 术语 | 大家常说 | 实际含义 |
|------|----------------|----------------------|
| 提示缓存（Prompt caching） | “缓存系统提示词” | 提供商级缓存：重复出现的提示词前缀会获得折扣（Anthropic 90%，OpenAI 50%）——OpenAI 无需改代码，Anthropic 需要显式标记 |
| 语义缓存（Semantic caching） | “智能缓存” | 为查询生成嵌入，计算其与历史查询的相似度，并在超过阈值时返回缓存响应——能捕获精确匹配无法识别的改写表达 |
| 精确缓存（Exact caching） | “哈希缓存” | 对完整提示词（model + messages + temperature）做哈希，并在输入完全一致时返回缓存响应——只适用于 temperature=0 的确定性调用 |
| 令牌桶（Token bucket） | “速率限制器” | 一种算法：每个用户拥有一个含 N 个 token 的桶，并按每秒 R 个 token 回填——允许最多到 N 的突发流量，同时约束平均速率为 R |
| 模型路由（Model routing） | “省钱路由” | 使用分类器把简单查询发送到便宜模型（GPT-4o-mini、Haiku），把复杂查询发送到昂贵模型（GPT-4o、Opus）——可节省 40-70% 的模型成本 |
| 成本追踪（Cost tracking） | “计量” | 记录每一次 API 调用的模型、tokens、延迟、成本和用户 ID，这样你就能准确知道钱花在哪里，以及哪些功能最贵 |
| 电路断路器（Circuit breaker） | “紧急开关” | 当支出接近预算上限时，自动让服务退化（更便宜的模型、仅缓存）或彻底停止请求 |
| Batch API | “批量折扣” | OpenAI 提供的异步处理能力，统一 50% 折扣——最多提交 50,000 个请求，并在 24 小时内得到结果 |
| 提示压缩（Prompt compression） | “token 节食” | 重写系统提示词与上下文，在保留含义的同时使用更少 tokens——更短的提示词更便宜，而且往往效果更好 |
| 缓存命中率（Cache hit rate） | “缓存效率” | 由缓存而非 LLM 返回结果的请求比例——生产环境聊天机器人常见为 40-60%，成本节省通常与之近似成正比 |

## 延伸阅读

- [Anthropic Prompt Caching Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) -- Anthropic 显式 `cache_control` 标记、定价和缓存生命周期行为的官方文档
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) -- OpenAI 的自动缓存、如何通过 usage 字段验证缓存命中，以及最小前缀长度
- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) -- 异步处理可获得 50% 折扣、JSONL 格式、24 小时完成窗口以及 50K 请求上限
- [GPTCache](https://github.com/zilliztech/GPTCache) -- 开源语义缓存库，支持多种嵌入后端、向量存储和淘汰策略
- [Martian Model Router](https://docs.withmartian.com) -- 生产级模型路由，可自动选择能处理每个查询且成本最低的模型
- [Not Diamond](https://www.notdiamond.ai) -- 基于 ML 的模型路由器，可从你的流量模式中学习，以优化跨提供商的成本/质量权衡
- [Helicone](https://www.helicone.ai) -- LLM 可观测性平台，作为代理层提供成本追踪、缓存、速率限制和预算告警
- [Dean & Barroso, "The Tail at Scale" (CACM 2013)](https://research.google/pubs/the-tail-at-scale/) -- 关于延迟、吞吐、TTFT/TPOT 百分位以及 hedged requests；解释“选择仍能满足 P95 的最便宜模型”背后的成本模型
- [Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)](https://arxiv.org/abs/2309.06180) -- vLLM 论文；解释为何 paged KV-cache + continuous batching 能让吞吐比朴素服务器高 24×，这正是“缓存与成本”之下的基础设施层
- [Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (ICLR 2024)](https://arxiv.org/abs/2307.08691) -- 与提示缓存正交的内核级降本方案；可与 speculative decoding 和 GQA 一起阅读，以完整理解成本曲线。
