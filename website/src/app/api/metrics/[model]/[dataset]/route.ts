import { NextRequest, NextResponse } from 'next/server';

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ model: string; dataset: string }> }
) {
  const { model, dataset } = await params;
  
  try {
    const response = await fetch(`${ML_SERVICE_URL}/metrics/${model}/${dataset}`);
    
    if (!response.ok) {
      return NextResponse.json({ error: 'Failed to get metrics' }, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Metrics API error:', error);
    return NextResponse.json({ error: 'ML service unavailable' }, { status: 503 });
  }
}
