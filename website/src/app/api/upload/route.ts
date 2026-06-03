import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    const text = await file.text();
    const lines = text.split(/\r?\n/).filter(line => line.trim() !== '');
    if (lines.length < 2) {
      return NextResponse.json({ error: 'CSV file must contain a header row and data rows' }, { status: 400 });
    }

    // Parse header channels
    const channels = lines[0].split(',').map(ch => ch.trim());
    const signals: number[][] = channels.map(() => []);

    // Parse time sample rows
    for (let i = 1; i < lines.length; i++) {
      const row = lines[i].split(',').map(val => parseFloat(val.trim()));
      if (row.length !== channels.length) continue;
      
      for (let chIdx = 0; chIdx < channels.length; chIdx++) {
        const val = row[chIdx];
        signals[chIdx].push(isNaN(val) ? 0.0 : val);
      }
    }

    const num_samples = signals[0]?.length || 0;
    const sampling_rate = 128;
    const duration_sec = Math.max(1, Math.round(num_samples / sampling_rate));

    return NextResponse.json({
      filename: file.name,
      channels,
      sampling_rate,
      duration_sec,
      num_samples,
      signals
    });
  } catch (error) {
    console.error('Upload API error:', error);
    return NextResponse.json({ error: 'Failed to parse EEG recording CSV file' }, { status: 500 });
  }
}
