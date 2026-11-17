export async function probeRuntime(request: (signal: AbortSignal) => Promise<Response>, parent?: AbortSignal, timeoutMs = 5000) {
  const controller = new AbortController();
  const cancel = () => controller.abort();
  parent?.addEventListener('abort', cancel, { once: true });
  if (parent?.aborted) cancel();
  const timeout = setTimeout(cancel, timeoutMs);
  try {
    const response = await request(controller.signal);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (data.name !== 'SuperImagem') throw new Error('Servidor inesperado');
    return { status: data.engineAvailable && data.gpu?.name ? 'ready' as const : 'no-gpu' as const, data };
  } catch {
    return { status: 'offline' as const, data: null };
  } finally {
    clearTimeout(timeout);
    parent?.removeEventListener('abort', cancel);
  }
}
