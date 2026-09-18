import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  ConnectionState,
  FrameAnalysis,
  HelloMessage,
  LiveMessage,
  ModelInfo,
  PipelineStage,
  StatusMessage,
} from '../types';

const API_BASE = import.meta.env.VITE_API_BASE ?? '';

function websocketUrl(): string {
  if (API_BASE) {
    return API_BASE.replace(/^http/, 'ws') + '/ws/live';
  }
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${protocol}://${window.location.host}/ws/live`;
}

interface LiveState {
  connection: ConnectionState;
  analysis: FrameAnalysis | null;
  hello: HelloMessage | null;
  models: ModelInfo[];
  stages: PipelineStage[];
  cameraConnected: boolean;
  cameraError: string;
  lastMessageAt: number | null;
}

const INITIAL: LiveState = {
  connection: 'connecting',
  analysis: null,
  hello: null,
  models: [],
  stages: [],
  cameraConnected: false,
  cameraError: '',
  lastMessageAt: null,
};

/** Subscribes to /ws/live and keeps the newest analysis frame. */
export function useLiveAnalysis() {
  const [state, setState] = useState<LiveState>(INITIAL);
  const socketRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(0);
  const timerRef = useRef<number | null>(null);
  const closedByUs = useRef(false);

  const connect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.close();
    }
    setState((previous) => ({
      ...previous,
      connection: retryRef.current === 0 ? 'connecting' : 'reconnecting',
    }));

    const socket = new WebSocket(websocketUrl());
    socketRef.current = socket;

    socket.onopen = () => {
      retryRef.current = 0;
      setState((previous) => ({ ...previous, connection: 'live' }));
    };

    socket.onmessage = (event) => {
      let message: LiveMessage;
      try {
        message = JSON.parse(event.data as string);
      } catch {
        return;
      }
      const now = Date.now();

      if (message.type === 'hello') {
        const hello = message as HelloMessage;
        setState((previous) => ({
          ...previous,
          hello,
          models: hello.models,
          stages: hello.stages,
          lastMessageAt: now,
        }));
        return;
      }
      if (message.type === 'status') {
        const status = message as StatusMessage;
        setState((previous) => ({
          ...previous,
          stages: status.stages,
          cameraConnected: status.camera_connected,
          cameraError: status.last_error,
          lastMessageAt: now,
        }));
        return;
      }
      const analysis = message as FrameAnalysis;
      setState((previous) => ({
        ...previous,
        analysis,
        stages: analysis.stages,
        cameraConnected: analysis.camera_connected,
        cameraError: analysis.camera_connected ? '' : previous.cameraError,
        lastMessageAt: now,
      }));
    };

    socket.onerror = () => {
      socket.close();
    };

    socket.onclose = () => {
      if (closedByUs.current) return;
      setState((previous) => ({ ...previous, connection: 'reconnecting' }));
      retryRef.current += 1;
      const delay = Math.min(1000 * 2 ** (retryRef.current - 1), 10000);
      timerRef.current = window.setTimeout(connect, delay);
    };
  }, []);

  useEffect(() => {
    closedByUs.current = false;
    connect();
    return () => {
      closedByUs.current = true;
      if (timerRef.current) window.clearTimeout(timerRef.current);
      socketRef.current?.close();
    };
  }, [connect]);

  // Mark the feed stale if nothing arrives for a while.
  useEffect(() => {
    const id = window.setInterval(() => {
      setState((previous) => {
        if (previous.connection !== 'live' || !previous.lastMessageAt) return previous;
        if (Date.now() - previous.lastMessageAt < 6000) return previous;
        return { ...previous, connection: 'offline' };
      });
    }, 2000);
    return () => window.clearInterval(id);
  }, []);

  return { ...state, reconnect: connect, apiBase: API_BASE };
}
