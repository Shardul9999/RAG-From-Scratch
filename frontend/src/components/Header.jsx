import React from 'react';
import { 
  Database, 
  Layers, 
  Cpu, 
  Activity, 
  FileText, 
  ShieldCheck, 
  RefreshCw 
} from 'lucide-react';

export default function Header({ 
  stats, 
  loadingStats, 
  onRefreshStats, 
  onOpenAuditLogs, 
  onToggleSidebar, 
  documentsCount,
  totalChunks
}) {
  const isPostgresOnline = stats?.postgres?.status === 'online';
  const isRedisOnline = stats?.redis?.status === 'online';

  return (
    <header className="border-b border-zinc-800/80 bg-zinc-950/80 backdrop-blur-md sticky top-0 z-30 px-4 lg:px-8 py-3.5">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        
        {/* Brand & Title */}
        <div className="flex items-center space-x-3">
          <div className="h-9 w-9 rounded-lg bg-zinc-900 border border-zinc-700/80 flex items-center justify-center shadow-inner">
            <Layers className="h-5 w-5 text-indigo-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-semibold tracking-tight text-zinc-100 m-0">
                Nexus RAG
              </h1>
              <span className="text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-zinc-700">
                Production Core
              </span>
            </div>
            <p className="text-xs text-zinc-500 m-0">
              pgvector • BGE Cross-Encoder • Gemini 2.5 Flash • Redis Cache
            </p>
          </div>
        </div>

        {/* Pipeline & Infrastructure Telemetry Badges */}
        <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
          {/* Postgres / pgvector status */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-zinc-900/90 border border-zinc-800 text-zinc-300">
            <span className={`inline-block h-2 w-2 rounded-full ${isPostgresOnline ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' : 'bg-rose-500'}`} />
            <Database className="h-3.5 w-3.5 text-zinc-400" />
            <span>Postgres</span>
            <span className="text-zinc-500">|</span>
            <span className="text-emerald-400 font-semibold">{totalChunks ?? stats?.postgres?.chunks_count ?? 0}</span>
            <span className="text-zinc-500 text-[10px]">chunks</span>
          </div>

          {/* Redis Semantic Cache status */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-zinc-900/90 border border-zinc-800 text-zinc-300">
            <span className={`inline-block h-2 w-2 rounded-full ${isRedisOnline ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' : 'bg-rose-500'}`} />
            <Cpu className="h-3.5 w-3.5 text-zinc-400" />
            <span>Redis Cache</span>
            <span className="text-zinc-500">|</span>
            <span className="text-indigo-400 font-semibold">{stats?.redis?.cached_entries ?? 0}</span>
            <span className="text-zinc-500 text-[10px]">keys</span>
          </div>

          {/* Refresh status button */}
          <button 
            onClick={onRefreshStats}
            title="Refresh system status"
            disabled={loadingStats}
            className="p-1.5 rounded-md bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 transition disabled:opacity-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loadingStats ? 'animate-spin text-indigo-400' : ''}`} />
          </button>

          {/* Divider */}
          <div className="h-4 w-px bg-zinc-800 hidden md:block" />

          {/* Audit Log Trigger */}
          <button
            onClick={onOpenAuditLogs}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-zinc-900 border border-zinc-800 text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100 transition"
          >
            <ShieldCheck className="h-3.5 w-3.5 text-indigo-400" />
            <span>Audit Log</span>
          </button>

          {/* Knowledge Base Toggle */}
          <button
            onClick={onToggleSidebar}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-indigo-600/10 border border-indigo-500/30 text-indigo-300 hover:bg-indigo-600/20 transition"
          >
            <FileText className="h-3.5 w-3.5 text-indigo-400" />
            <span>Documents ({documentsCount})</span>
          </button>
        </div>

      </div>
    </header>
  );
}
