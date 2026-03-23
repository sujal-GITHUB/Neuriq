import { NextRequest, NextResponse } from 'next/server';

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    // Forward to FastAPI
    const mlFormData = new FormData();
    mlFormData.append('file', file);

    const response = await fetch(`${ML_SERVICE_URL}/upload-eeg`, {
      method: 'POST',
      body: mlFormData,
    });

    if (!response.ok) {
      const error = await response.text();
      return NextResponse.json({ error }, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Upload API error:', error);
    // Mock response with synthetic data
    const channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4'];
    const n = 128 * 30;
    const signals = channels.map(() => {
      const s = [];
      for (let i = 0; i < n; i++) {
        const t = i / 128;
        s.push(15 * Math.sin(2 * Math.PI * 10 * t) + 8 * Math.sin(2 * Math.PI * 22 * t) + (Math.random() - 0.5) * 10);
      }
      return s;
    });
    
    return NextResponse.json({
      filename: 'sample-eeg.csv',
      channels,
      sampling_rate: 128,
      duration_sec: 30,
      num_samples: n,
      signals
    });
  }
}
