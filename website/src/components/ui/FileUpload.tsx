'use client';

import { useCallback, useState } from 'react';
import { motion } from 'framer-motion';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  accept?: string;
  isLoading?: boolean;
}

export default function FileUpload({ onFileSelect, accept = '.csv,.edf,.bdf,.dat', isLoading }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      setSelectedFile(file);
      onFileSelect(file);
    }
  }, [onFileSelect]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      onFileSelect(file);
    }
  }, [onFileSelect]);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={`
        relative border-2 border-dashed rounded-xl p-8 text-center transition-all cursor-pointer
        ${isDragging
          ? 'border-indigo-500 bg-indigo-500/10'
          : 'border-white/10 hover:border-indigo-500/50 hover:bg-white/[0.02]'
        }
        ${isLoading ? 'opacity-50 pointer-events-none' : ''}
      `}
    >
      <input
        type="file"
        accept={accept}
        onChange={handleChange}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        aria-label="Upload EEG file"
      />

      <div className="space-y-3">
        <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-indigo-500/20 to-cyan-500/20 flex items-center justify-center mx-auto">
          <svg className="w-7 h-7 text-indigo-400" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
          </svg>
        </div>

        {selectedFile ? (
          <div>
            <p className="text-sm font-medium text-white">{selectedFile.name}</p>
            <p className="text-xs text-slate-400">{(selectedFile.size / 1024).toFixed(1)} KB</p>
          </div>
        ) : (
          <div>
            <p className="text-sm text-slate-300">
              <span className="text-indigo-400 font-medium">Click to upload</span> or drag and drop
            </p>
            <p className="text-xs text-slate-500 mt-1">
              Supports .csv, .edf, .bdf, .dat files
            </p>
          </div>
        )}

        {isLoading && (
          <div className="flex items-center justify-center gap-2">
            <div className="w-4 h-4 border-2 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
            <span className="text-xs text-slate-400">Processing...</span>
          </div>
        )}
      </div>
    </motion.div>
  );
}
