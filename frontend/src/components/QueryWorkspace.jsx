import React, { useState, useRef, useEffect } from 'react';
import { 
  Search, 
  Zap, 
  Cpu, 
  Clock, 
  Copy, 
  Check, 
  Sparkles, 
  FileText, 
  ChevronDown, 
  ChevronUp, 
  ExternalLink,
  CornerDownLeft,
  ArrowRight,
  Database
} from 'lucide-react';

const SUGGESTIONS = [
  "What core technical skills are mentioned?",
  "Summarize the professional work experience",
  "What education and degrees are listed?",
  "List projects with technologies used"
];

export default function QueryWorkspace({ onQuerySubmitted }) {
  const [queryText, setQueryText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);
  const [activeCitation, setActiveCitation] = useState(null);
  const [expandedChunks, setExpandedChunks] = useState({});
  const citationRefs = useRef({});

  const handleSearch = async (queryToRun) => {
    const q = (queryToRun || queryText).trim();
    if (!q || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setActiveCitation(null);

    try {
      const res = await fetch(`/query?q=${encodeURIComponent(q)}`);
      if (!res.ok) {
        throw new Error(`Server returned error status ${res.status}`);
      }
      const data = await res.json();
      setResult(data);
      if (onQuerySubmitted) onQuerySubmitted();
    } catch (err) {
      setError(err.message || 'Failed to execute query');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSearch();
    }
  };

  const handleCopy = () => {
    if (!result?.answer) return;
    navigator.clipboard.writeText(result.answer);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const scrollToCitation = (chunkNum) => {
    setActiveCitation(chunkNum);
    const elem = citationRefs.current[chunkNum];
    if (elem) {
      elem.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  const toggleChunkExpand = (chunkNum) => {
    setExpandedChunks(prev => ({
      ...prev,
      [chunkNum]: !prev[chunkNum]
    }));
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      
      {/* Search Input Box */}
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/70 shadow-2xl p-2 focus-within:border-zinc-700 transition-all">
        <div className="flex items-center px-3 py-1">
          <Search className={`h-5 w-5 mr-3 shrink-0 ${loading ? 'animate-pulse text-indigo-400' : 'text-zinc-500'}`} />
          <input
            type="text"
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question against indexed documents..."
            className="w-full bg-transparent border-none text-zinc-100 placeholder-zinc-500 text-sm focus:outline-none focus:ring-0"
          />
          {queryText && (
            <button
              onClick={() => setQueryText('')}
              className="text-xs text-zinc-500 hover:text-zinc-300 mr-2 px-1.5 py-0.5 rounded"
            >
              Clear
            </button>
          )}
          <button
            onClick={() => handleSearch()}
            disabled={!queryText.trim() || loading}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:bg-zinc-800 disabled:text-zinc-600 text-white text-xs font-medium transition shrink-0"
          >
            <span>Ask</span>
            <CornerDownLeft className="h-3 w-3" />
          </button>
        </div>

        {/* Suggestion Chips */}
        <div className="pt-2 px-3 pb-1 border-t border-zinc-800/60 flex flex-wrap items-center gap-1.5">
          <span className="text-[10px] font-mono uppercase text-zinc-500 mr-1">Suggestions:</span>
          {SUGGESTIONS.map((item, idx) => (
            <button
              key={idx}
              onClick={() => {
                setQueryText(item);
                handleSearch(item);
              }}
              className="text-[11px] px-2 py-0.5 rounded-md bg-zinc-800/70 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 border border-zinc-700/60 transition"
            >
              {item}
            </button>
          ))}
        </div>
      </div>

      {/* Loading Pipeline State */}
      {loading && (
        <div className="rounded-xl border border-zinc-800 bg-zinc-950/60 p-6 space-y-4 text-center">
          <div className="inline-flex items-center justify-center p-3 rounded-xl bg-indigo-950/30 border border-indigo-900/50 text-indigo-400 animate-pulse">
            <Cpu className="h-6 w-6" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-zinc-200 m-0">Executing RAG Pipeline</h3>
            <p className="text-xs text-zinc-500 mt-1 m-0">
              Query Embedding → pgvector Similarity Search → Cross-Encoder Rerank → Gemini Synthesis
            </p>
          </div>
          <div className="max-w-md mx-auto h-1.5 bg-zinc-800 rounded-full overflow-hidden">
            <div className="h-full bg-gradient-to-r from-indigo-500 to-emerald-400 rounded-full animate-[shimmer_2s_infinite]" />
          </div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="p-4 rounded-xl border border-rose-900/80 bg-rose-950/20 text-rose-300 text-xs flex items-start gap-3">
          <div className="p-1 rounded bg-rose-900/40 text-rose-400 shrink-0">!</div>
          <div>
            <p className="font-semibold text-rose-200 m-0">Error processing query</p>
            <p className="mt-0.5 m-0 opacity-90">{error}</p>
          </div>
        </div>
      )}

      {/* Query Result Card */}
      {result && (
        <div className="rounded-xl border border-zinc-800 bg-zinc-950 shadow-2xl overflow-hidden divide-y divide-zinc-800/80">
          
          {/* Metadata & Source Header */}
          <div className="p-4 bg-zinc-900/40 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              {result.source === 'semantic_cache' ? (
                <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-cyan-950/60 border border-cyan-800 text-cyan-300 text-xs font-mono">
                  <Zap className="h-3.5 w-3.5 text-cyan-400" />
                  <span>Semantic Cache Hit</span>
                </div>
              ) : (
                <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-950/60 border border-emerald-800 text-emerald-300 text-xs font-mono">
                  <Sparkles className="h-3.5 w-3.5 text-emerald-400" />
                  <span>Cross-Encoder Reranked + LLM</span>
                </div>
              )}

              {/* Latency badge */}
              {result.latency_ms !== undefined && (
                <div className="inline-flex items-center gap-1 text-[11px] font-mono text-zinc-400 bg-zinc-900 px-2 py-1 rounded border border-zinc-800">
                  <Clock className="h-3 w-3 text-zinc-500" />
                  <span>{result.latency_ms} ms</span>
                </div>
              )}
            </div>

            {/* Token details or Copy button */}
            <div className="flex items-center gap-2">
              {result.tokens && (
                <div className="hidden sm:flex items-center gap-2 text-[11px] font-mono text-zinc-500">
                  <span>Tokens: <strong className="text-zinc-300">{result.tokens.input}</strong> in / <strong className="text-zinc-300">{result.tokens.output}</strong> out</span>
                  {result.tokens.cost > 0 && (
                    <span className="text-emerald-400 font-semibold">${Number(result.tokens.cost).toFixed(6)}</span>
                  )}
                </div>
              )}

              <button
                onClick={handleCopy}
                className="flex items-center gap-1 px-2.5 py-1 rounded bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 text-xs transition"
              >
                {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
            </div>
          </div>

          {/* Generated Answer */}
          <div className="p-6">
            <h4 className="text-xs uppercase tracking-wider font-mono text-zinc-500 mb-3 m-0">Answer</h4>
            <div className="prose prose-invert max-w-none text-zinc-100 text-sm leading-relaxed whitespace-pre-line font-sans">
              {result.answer}
            </div>
          </div>

          {/* Citations & Evidence Chunks */}
          {result.citations && result.citations.length > 0 && (
            <div className="p-6 bg-zinc-900/20 space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-xs uppercase tracking-wider font-mono text-zinc-400 flex items-center gap-1.5 m-0">
                  <FileText className="h-3.5 w-3.5 text-indigo-400" />
                  <span>Supporting Evidence & Citations ({result.citations.length})</span>
                </h4>
                <span className="text-[11px] text-zinc-500 font-mono">
                  Verified against pgvector chunks
                </span>
              </div>

              <div className="grid grid-cols-1 gap-2.5">
                {result.citations.map((c, i) => {
                  const isHighlighted = activeCitation === c.chunk;
                  const isExpanded = expandedChunks[c.chunk];
                  const fileName = c.source ? c.source.replace(/^temp\//, '') : 'Unknown file';

                  return (
                    <div
                      key={i}
                      ref={el => citationRefs.current[c.chunk] = el}
                      className={`rounded-lg border p-3 transition-all ${
                        isHighlighted
                          ? 'border-indigo-500 bg-indigo-950/30 ring-1 ring-indigo-500/50'
                          : 'border-zinc-800/80 bg-zinc-900/50 hover:bg-zinc-900/80'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2 mb-2">
                        <div className="flex items-center gap-2">
                          <span className="h-5 w-5 rounded bg-indigo-600/20 border border-indigo-500/40 text-indigo-300 font-mono text-xs flex items-center justify-center font-bold">
                            #{c.chunk}
                          </span>
                          <span className="text-xs font-medium text-zinc-200">
                            {fileName}
                          </span>
                        </div>

                        <div className="flex items-center gap-2">
                          {c.score !== undefined && c.score !== null && (
                            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-800 text-emerald-400 border border-zinc-700">
                              score: {Number(c.score).toFixed(4)}
                            </span>
                          )}
                          {c.content && (
                            <button
                              onClick={() => toggleChunkExpand(c.chunk)}
                              className="text-zinc-500 hover:text-zinc-300 p-0.5 rounded"
                            >
                              {isExpanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                            </button>
                          )}
                        </div>
                      </div>

                      {/* Content snippet */}
                      {c.content && (
                        <p className={`text-xs text-zinc-400 font-mono bg-zinc-950/70 p-2.5 rounded border border-zinc-800/60 leading-normal m-0 ${isExpanded ? '' : 'line-clamp-2'}`}>
                          {c.content}
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

        </div>
      )}

      {/* Empty State / Pipeline Architecture Diagram */}
      {!result && !loading && (
        <div className="border border-zinc-800/70 bg-zinc-950/40 rounded-xl p-8 text-center space-y-6">
          <div className="max-w-md mx-auto space-y-2">
            <h3 className="text-base font-medium text-zinc-200 m-0">
              High-Precision Technical RAG
            </h3>
            <p className="text-xs text-zinc-500 leading-relaxed m-0">
              Upload a PDF document to begin. Queries hit the semantic cache first; on miss, candidate chunks are retrieved from PostgreSQL via pgvector cosine distance, reranked with BGE Cross-Encoder, and synthesized into a cited answer using Gemini 2.5 Flash.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 max-w-2xl mx-auto pt-2 text-left font-mono text-xs">
            <div className="p-3 rounded-lg border border-zinc-800 bg-zinc-900/40">
              <span className="text-[10px] text-zinc-500 uppercase block mb-1">01. Cache</span>
              <p className="text-zinc-300 font-semibold m-0">Redis</p>
              <p className="text-[10px] text-zinc-500 mt-1 m-0">Cosine sim &gt; 0.90</p>
            </div>
            <div className="p-3 rounded-lg border border-zinc-800 bg-zinc-900/40">
              <span className="text-[10px] text-zinc-500 uppercase block mb-1">02. Search</span>
              <p className="text-zinc-300 font-semibold m-0">pgvector</p>
              <p className="text-[10px] text-zinc-500 mt-1 m-0">3072d Top 20</p>
            </div>
            <div className="p-3 rounded-lg border border-zinc-800 bg-zinc-900/40">
              <span className="text-[10px] text-zinc-500 uppercase block mb-1">03. Rerank</span>
              <p className="text-zinc-300 font-semibold m-0">BGE Base</p>
              <p className="text-[10px] text-zinc-500 mt-1 m-0">Cross-Encoder Top 5</p>
            </div>
            <div className="p-3 rounded-lg border border-zinc-800 bg-zinc-900/40">
              <span className="text-[10px] text-zinc-500 uppercase block mb-1">04. Synthesis</span>
              <p className="text-zinc-300 font-semibold m-0">Gemini 2.5</p>
              <p className="text-[10px] text-zinc-500 mt-1 m-0">Cited JSON schema</p>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
