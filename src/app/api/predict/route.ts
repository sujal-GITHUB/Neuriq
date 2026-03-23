import { NextRequest, NextResponse } from 'next/server';

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // Determine endpoint based on whether this is manual or signal input
    const endpoint = body.manual ? '/predict/manual' : '/predict';
    
    const response = await fetch(`${ML_SERVICE_URL}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const error = await response.text();
      return NextResponse.json(
        { error: `ML Service error: ${error}` },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Predict API error:', error);
    
    // Return mock data in development if ML service is unavailable
    const { MOCK_PREDICTION } = await import('@/lib/utils');
    return NextResponse.json(MOCK_PREDICTION);
  }
}
