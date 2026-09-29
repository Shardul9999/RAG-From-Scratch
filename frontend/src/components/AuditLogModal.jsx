import React, { useEffect, useState } from 'react';
import { ShieldCheck, X, RefreshCw, Clock, DollarSign, Layers } from 'lucide-react';

export default function AuditLogModal({ isOpen, onClose }) {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchLogs = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/audit-logs');
      if (res.ok) {
        const data = await res.json();
        setLogs(data);
      }
    } catch (err) {
      console.error('Failed to load audit logs:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchLogs();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
      <div className="bg-zinc-950 border border-zinc-800 rounded-xl w-full max-w-4xl max-h-[85vh] flex flex-col shadow-2xl overflow-hidden">
        
        {/* Modal Header */}
        <div className="p-4 border-b border-zinc-800 flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <div className="p-2 rounded-lg bg-zinc-900 border border-zinc-700">
              <ShieldCheck className="h-4 w-4 text-indigo-400" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-zinc-100 tracking-tight m-0">
                AI Pipeline Audit Logs
              </h2>
              <p className="text-xs text-zinc-500 m-0">
                PostgreSQL table: <code className="text-zinc-400 font-mono text-[11px]">public.ai_audit_log</code>
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={fetchLogs}
              disabled={loading}
              title="Refresh logs"
              className="p-1.5 rounded-md bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-zinc-200 transition"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-md text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 transition"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Logs Table */}
        <div className="flex-1 overflow-y-auto p-4">
          {logs.length === 0 ? (
            <div className="text-center py-12 text-zinc-500 text-sm">
              {loading ? 'Fetching audit logs...' : 'No audit records found yet. Run queries to generate logs.'}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-zinc-800 text-zinc-500 uppercase font-mono text-[10px]">
                    <th className="py-2.5 px-3">Timestamp</th>
                    <th className="py-2.5 px-3">Query</th>
                    <th className="py-2.5 px-3">Model</th>
                    <th className="py-2.5 px-3 text-right">Tokens (In / Out)</th>
                    <th className="py-2.5 px-3 text-right">Cost ($)</th>
                    <th className="py-2.5 px-3 text-right">Chunks</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/60 font-mono">
                  {logs.map((log) => (
                    <tr key={log.id} className="hover:bg-zinc-900/40 transition">
                      <td className="py-3 px-3 text-zinc-400 whitespace-nowrap text-[11px]">
                        {log.created_at ? new Date(log.created_at).toLocaleString() : 'N/A'}
                      </td>
                      <td className="py-3 px-3 text-zinc-200 font-sans max-w-xs truncate font-medium">
                        {log.query}
                      </td>
                      <td className="py-3 px-3 text-indigo-400 whitespace-nowrap text-[11px]">
                        {log.model}
                      </td>
                      <td className="py-3 px-3 text-right text-zinc-300 whitespace-nowrap text-[11px]">
                        <span className="text-zinc-400">{log.input_tokens || 0}</span>
                        <span className="text-zinc-600"> / </span>
                        <span className="text-indigo-300">{log.output_tokens || 0}</span>
                      </td>
                      <td className="py-3 px-3 text-right text-emerald-400 whitespace-nowrap text-[11px]">
                        ${Number(log.cost || 0).toFixed(6)}
                      </td>
                      <td className="py-3 px-3 text-right text-zinc-400 whitespace-nowrap text-[11px]">
                        {log.chunks_used || 0}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="p-3 border-t border-zinc-800 bg-zinc-900/30 flex items-center justify-between text-xs text-zinc-500 font-mono">
          <span>Showing latest {logs.length} transactions</span>
          <button
            onClick={onClose}
            className="px-3 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-200 transition"
          >
            Close
          </button>
        </div>

      </div>
    </div>
  );
}
