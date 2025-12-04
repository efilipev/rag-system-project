import { useState, useRef } from 'react';
import { Button } from 'react-aria-components';
import { Upload, File, X, Loader2 } from 'lucide-react';
import { UploadedDocument } from '@/types';
import { apiClient } from '@/lib/api';
import { formatFileSize } from '@/lib/utils';
import { cn } from '@/lib/utils';

interface DocumentUploadProps {
  sessionId: string;
  currentDocument?: UploadedDocument;
  onDocumentUploaded: (document: UploadedDocument) => void;
  onDocumentRemoved: () => void;
}

export function DocumentUpload({
  sessionId,
  currentDocument,
  onDocumentUploaded,
  onDocumentRemoved,
}: DocumentUploadProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = async (file: File) => {
    if (!file.type.includes('pdf')) {
      setError('Please upload a PDF file');
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      setError('File size must be less than 50MB');
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      const document = await apiClient.uploadDocument(file, sessionId);
      onDocumentUploaded(document);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  const handleRemove = () => {
    onDocumentRemoved();
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  if (currentDocument) {
    return (
      <div className="space-y-3">
        <div className="flex items-center justify-between rounded-xl border border-slate-200 dark:border-dark-800 bg-white dark:bg-dark-850 p-3 shadow-sm dark:shadow-none">
          <div className="flex items-center space-x-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-50 dark:bg-primary-500/10 ring-1 ring-primary-100 dark:ring-primary-500/20">
              <File className="h-5 w-5 text-primary-600 dark:text-primary-400" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-slate-900 dark:text-white truncate">
                {currentDocument.name}
              </p>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                {formatFileSize(currentDocument.size)}
              </p>
            </div>
          </div>
          <Button
            onPress={handleRemove}
            className="flex h-8 w-8 items-center justify-center rounded-lg hover:bg-slate-100 dark:hover:bg-dark-800 transition-colors"
          >
            <X className="h-4 w-4 text-slate-500 dark:text-slate-400" />
          </Button>
        </div>

        {currentDocument.base64 && (
          <div className="rounded-xl border border-slate-200 dark:border-dark-800 bg-white dark:bg-dark-850 overflow-hidden shadow-sm dark:shadow-none">
            <iframe
              src={`data:application/pdf;base64,${currentDocument.base64}`}
              className="w-full h-[600px]"
              title="PDF Preview"
            />
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div
        className={cn(
          'relative rounded-xl border-2 border-dashed transition-all',
          dragActive
            ? 'border-primary-500 dark:border-primary-500 bg-primary-50 dark:bg-primary-500/10 scale-102'
            : 'border-slate-300 dark:border-dark-700 bg-white dark:bg-dark-850/50 hover:border-slate-400 dark:hover:border-dark-600',
          isUploading && 'pointer-events-none opacity-50'
        )}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          onChange={handleChange}
          className="hidden"
        />

        <div className="flex flex-col items-center justify-center px-6 py-10">
          {isUploading ? (
            <>
              <Loader2 className="h-12 w-12 animate-spin text-primary-500 dark:text-primary-400" />
              <p className="mt-4 text-sm font-medium text-slate-600 dark:text-slate-400">Uploading document...</p>
            </>
          ) : (
            <>
              <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-primary-50 dark:bg-primary-500/10 mb-4 ring-1 ring-primary-100 dark:ring-primary-500/20">
                <Upload className="h-7 w-7 text-primary-600 dark:text-primary-400" />
              </div>
              <p className="text-sm font-bold text-slate-900 dark:text-white">
                Upload a PDF document
              </p>
              <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                Drag and drop or click to browse
              </p>
              <Button
                onPress={handleButtonClick}
                className="mt-4 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 px-5 py-2.5 text-sm font-semibold text-white hover:shadow-lg hover:shadow-primary-500/30 transition-all hover:scale-105"
              >
                Select File
              </Button>
              <p className="mt-3 text-xs text-slate-400 dark:text-slate-500">PDF up to 50MB</p>
            </>
          )}
        </div>
      </div>

      {error && (
        <div className="rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800/50 p-3 shadow-sm dark:shadow-none">
          <p className="text-sm font-medium text-red-800 dark:text-red-300">{error}</p>
        </div>
      )}
    </div>
  );
}
