import React, { useState, useRef } from 'react';
import { 
  UploadCloud, 
  FileText, 
  Trash2, 
  RefreshCw, 
  Database, 
  X, 
  AlertTriangle,
  CheckCircle,
  FileCheck
} from 'lucide-react';

export default function KnowledgeBase({ 
  isOpen, 
  onClose, 
  documents, 
  loadingDocs, 
  onRefreshDocs, 
  onDeleteDoc, 
  onFlushCache, 
  onFlushAllChunks 
}) {
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [confirmFlushDb, setConfirmFlushDb] = useState(false);
  const [confirmFlushCache, setConfirmFlushCache] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  };

  const handleFileUpload = async (file) => {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploadStatus({ type: 'error', message: 'Only PDF documents are supported' });
      return;
    }

    setUploading(true);
    setUploadStatus({ type: 'info', message: `Uploading & dispatching ${file.name}...` });

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Upload failed with status ${res.status}`);
      }

      const data = await res.json();
      setUploadStatus({ 
        type: 'success', 
        message: `Dispatched to Celery worker! Chunks will appear in database in a few seconds.` 
      });

      // Poll document list refresh after a short delay
      setTimeout(() => {
        onRefreshDocs();
      }, 3000);
      setTimeout(() => {
        onRefreshDocs();
      }, 6000);
    } catch (err) {
      setUploadStatus({ type: 'error', message: err.message || 'Failed to upload document' });
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  if (!isOpen) return null;

  return (
    <aside className="fixed inset-y-0 right-0 z-40 w-full sm:w-96 bg-zinc-950 border-l border-zinc-800 shadow-2xl flex flex-col transform transition-transform duration-200">
      
      {/* Header */}
      <div className="p-4 border-b border-zinc-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FileCheck className="h-5 w-5 text-indigo-400" />
          <h2 className="text-sm font-semibold text-zinc-100 tracking-tight m-0">
            Knowledge Base
          </h2>
        </div>
        <button 
          onClick={onClose}
          className="p-1.5 rounded-md text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 transition"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-5">
        
        {/* Dropzone Upload */}
        <div>
          <label className="text-xs font-medium uppercase tracking-wider text-zinc-400 block mb-2 font-mono">
            Ingest Document
          </label>
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border border-dashed rounded-lg p-5 text-center cursor-pointer transition-all ${
              dragActive 
                ? 'border-indigo-400 bg-indigo-950/20' 
                : 'border-zinc-800 hover:border-zinc-700 bg-zinc-900/40 hover:bg-zinc-900/70'
            }`}
          >
            <input 
              ref={fileInputRef}
              type="file" 
              accept=".pdf" 
              className="hidden" 
              onChange={handleFileInput} 
            />
            <UploadCloud className={`h-8 w-8 mx-auto mb-2 ${uploading ? 'animate-bounce text-indigo-400' : 'text-zinc-500'}`} />
            <p className="text-xs font-medium text-zinc-200 m-0">
              {uploading ? 'Uploading to pipeline...' : 'Drop PDF here or click to browse'}
            </p>
            <p className="text-[11px] text-zinc-500 mt-1 m-0">
              Automatic text extraction • 150w chunks • 3072d Gemini embedding
            </p>
          </div>

          {/* Status Alert */}
          {uploadStatus && (
            <div className={`mt-2.5 p-2.5 rounded-md text-xs border ${
              uploadStatus.type === 'error' 
                ? 'bg-rose-950/40 border-rose-800/80 text-rose-300' 
                : uploadStatus.type === 'success'
                ? 'bg-emerald-950/40 border-emerald-800/80 text-emerald-300'
                : 'bg-zinc-900 border-zinc-700 text-zinc-300'
            }`}>
              {uploadStatus.message}
            </div>
          )}
        </div>

        {/* Indexed Documents */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-xs font-medium uppercase tracking-wider text-zinc-400 font-mono">
              Indexed Files ({documents.length})
            </label>
            <button 
              onClick={onRefreshDocs}
              disabled={loadingDocs}
              title="Refresh document list"
              className="p-1 text-zinc-500 hover:text-zinc-300 transition"
            >
              <RefreshCw className={`h-3 w-3 ${loadingDocs ? 'animate-spin' : ''}`} />
            </button>
          </div>

          {documents.length === 0 ? (
            <div className="rounded-lg border border-zinc-800/80 bg-zinc-900/30 p-4 text-center">
              <FileText className="h-6 w-6 text-zinc-600 mx-auto mb-1.5" />
              <p className="text-xs text-zinc-400 font-medium m-0">No documents indexed</p>
              <p className="text-[11px] text-zinc-500 mt-0.5 m-0">Upload a PDF to start answering questions</p>
            </div>
          ) : (
            <div className="space-y-2">
              {documents.map((doc, idx) => {
                const displayName = doc.source.replace(/^temp\//, '');
                return (
                  <div 
                    key={idx}
                    className="p-3 rounded-lg border border-zinc-800/80 bg-zinc-900/60 hover:bg-zinc-900 transition flex items-center justify-between group"
                  >
                    <div className="flex items-center space-x-2.5 min-w-0 pr-2">
                      <div className="h-7 w-7 rounded bg-zinc-800 flex items-center justify-center shrink-0">
                        <FileText className="h-4 w-4 text-indigo-400" />
                      </div>
                      <div className="truncate">
                        <p className="text-xs font-medium text-zinc-200 truncate m-0">
                          {displayName}
                        </p>
                        <p className="text-[10px] text-zinc-500 font-mono m-0">
                          {doc.chunks} chunks stored
                        </p>
                      </div>
                    </div>

                    <button
                      onClick={() => onDeleteDoc(doc.source)}
                      title={`Delete ${displayName}`}
                      className="p-1.5 text-zinc-500 hover:text-rose-400 rounded hover:bg-zinc-800 opacity-60 group-hover:opacity-100 transition"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Maintenance / Cache Management */}
        <div className="pt-4 border-t border-zinc-800/80 space-y-3">
          <label className="text-xs font-medium uppercase tracking-wider text-zinc-400 block font-mono">
            Pipeline Actions
          </label>

          {/* Flush Semantic Cache */}
          <div>
            {confirmFlushCache ? (
              <div className="p-2.5 rounded-lg border border-amber-900/60 bg-amber-950/20 text-xs space-y-2">
                <p className="text-amber-300 font-medium m-0">Clear all cached queries in Redis?</p>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => {
                      onFlushCache();
                      setConfirmFlushCache(false);
                    }}
                    className="px-2 py-1 bg-amber-600 hover:bg-amber-500 text-zinc-950 font-semibold rounded text-[11px] transition"
                  >
                    Confirm Clear
                  </button>
                  <button
                    onClick={() => setConfirmFlushCache(false)}
                    className="px-2 py-1 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded text-[11px] transition"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <button
                onClick={() => setConfirmFlushCache(true)}
                className="w-full text-left px-3 py-2 rounded-md border border-zinc-800 bg-zinc-900/50 hover:bg-zinc-900 text-xs text-zinc-300 flex items-center justify-between transition"
              >
                <span>Clear Semantic Cache</span>
                <span className="text-[10px] text-zinc-500 font-mono">Redis</span>
              </button>
            )}
          </div>

          {/* Flush All Chunks */}
          <div>
            {confirmFlushDb ? (
              <div className="p-2.5 rounded-lg border border-rose-900/60 bg-rose-950/20 text-xs space-y-2">
                <p className="text-rose-300 font-medium m-0">
                  Delete ALL chunks from pgvector table?
                </p>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => {
                      onFlushAllChunks();
                      setConfirmFlushDb(false);
                    }}
                    className="px-2 py-1 bg-rose-600 hover:bg-rose-500 text-zinc-950 font-semibold rounded text-[11px] transition"
                  >
                    Confirm Truncate
                  </button>
                  <button
                    onClick={() => setConfirmFlushDb(false)}
                    className="px-2 py-1 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded text-[11px] transition"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <button
                onClick={() => setConfirmFlushDb(true)}
                className="w-full text-left px-3 py-2 rounded-md border border-zinc-800 bg-zinc-900/50 hover:bg-zinc-900 text-xs text-rose-400/90 hover:text-rose-300 flex items-center justify-between transition"
              >
                <span>Truncate Chunks Table</span>
                <span className="text-[10px] text-zinc-500 font-mono">Postgres</span>
              </button>
            )}
          </div>

        </div>

      </div>

      <div className="p-3 border-t border-zinc-800 bg-zinc-950 text-[10px] font-mono text-zinc-500 text-center">
        Tenant: <span className="text-zinc-300">default</span> • Dimension: <span className="text-zinc-300">3072</span>
      </div>

    </aside>
  );
}
