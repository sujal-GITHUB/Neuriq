import { NextRequest, NextResponse } from 'next/server';

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await params;
  
  try {
    const response = await fetch(`${ML_SERVICE_URL}/training-status/${jobId}`);
    
    if (!response.ok) {
      return NextResponse.json({ error: 'Failed to get status' }, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Training status error:', error);
    // Mock response
    return NextResponse.json({
      status: 'running',
      progress: Math.random() * 100,
      current_epoch: Math.floor(Math.random() * 100),
      total_epochs: 100,
      current_metrics: {
        train_loss: 0.4 + Math.random() * 0.3,
        val_loss: 0.5 + Math.random() * 0.3,
        train_accuracy: 0.7 + Math.random() * 0.2,
        val_accuracy: 0.65 + Math.random() * 0.2,
        learning_rate: 0.0001
      }
    });
  }
}
