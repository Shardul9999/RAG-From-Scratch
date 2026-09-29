import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import QueryWorkspace from './components/QueryWorkspace';
import KnowledgeBase from './components/KnowledgeBase';
import AuditLogModal from './components/AuditLogModal';

export default function App() {
  const [stats, setStats] = useState(null);
  const [loadingStats, setLoadingStats] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isAuditModalOpen, setIsAuditModalOpen] = useState(false);

  const fetchStats = async () => {
    setLoadingStats(true);
    try {
      const res = await fetch('/api/stats');
      if (res.ok) {
        const data = await res.json();
        setStats(data);
      }
    } catch (err) {
      console.error('Failed to load stats:', err);
    } finally {
      setLoadingStats(false);
    }
  };

  const fetchDocuments = async () => {
    setLoadingDocs(true);
    try {
      const res = await fetch('/api/documents');
      if (res.ok) {
        const data = await res.json();
        setDocuments(data);
      }
    } catch (err) {
      console.error('Failed to load documents:', err);
    } finally {
      setLoadingDocs(false);
    }
  };

  useEffect(() => {
    fetchStats();
    fetchDocuments();
    const interval = setInterval(() => {
      fetchStats();
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleDeleteDocument = async (source) => {
    try {
      const res = await fetch('/api/flush-chunks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source })
      });
      if (res.ok) {
        await fetchDocuments();
        await fetchStats();
      }
    } catch (err) {
      console.error('Failed to delete document:', err);
    }
  };

  const handleFlushCache = async () => {
    try {
      const res = await fetch('/api/flush-cache', { method: 'POST' });
      if (res.ok) {
        await fetchStats();
      }
    } catch (err) {
      console.error('Failed to flush cache:', err);
    }
  };

  const handleFlushAllChunks = async () => {
    try {
      const res = await fetch('/api/flush-chunks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      if (res.ok) {
        await fetchDocuments();
        await fetchStats();
      }
    } catch (err) {
      console.error('Failed to truncate chunks:', err);
    }
  };

  const totalChunks = documents.reduce((sum, d) => sum + (d.chunks || 0), 0);

  return (
    <div className="min-h-screen bg-[#090a0f] text-zinc-100 flex flex-col font-sans selection:bg-indigo-900/60 selection:text-indigo-200">
      
      {/* Navigation Header */}
      <Header
        stats={stats}
        loadingStats={loadingStats}
        onRefreshStats={() => {
          fetchStats();
          fetchDocuments();
        }}
        onOpenAuditLogs={() => setIsAuditModalOpen(true)}
        onToggleSidebar={() => setIsSidebarOpen(prev => !prev)}
        documentsCount={documents.length}
        totalChunks={totalChunks}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 lg:px-8 py-8">
        <QueryWorkspace
          onQuerySubmitted={() => {
            fetchStats();
          }}
        />
      </main>

      {/* Knowledge Base Sidebar Drawer */}
      <KnowledgeBase
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        documents={documents}
        loadingDocs={loadingDocs}
        onRefreshDocs={fetchDocuments}
        onDeleteDoc={handleDeleteDocument}
        onFlushCache={handleFlushCache}
        onFlushAllChunks={handleFlushAllChunks}
      />

      {/* Audit Log Modal */}
      <AuditLogModal
        isOpen={isAuditModalOpen}
        onClose={() => setIsAuditModalOpen(false)}
      />

      {/* Subtle Footer */}
      <footer className="border-t border-zinc-900 py-4 px-6 text-center text-[11px] font-mono text-zinc-600">
        Nexus RAG Engine • PostgreSQL 16 pgvector • BAAI/bge-reranker-base • Gemini 2.5 Flash
      </footer>

    </div>
  );
}
