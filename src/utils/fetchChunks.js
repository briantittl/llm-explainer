const CACHE_PREFIX = 'onnx-model-cache';
// The cache name includes a version taken from the model's manifest, so browsers
// automatically drop old model files and re-download when the model changes.
let CACHE_NAME = `${CACHE_PREFIX}-default`;

// Downloads at most this many chunks at once. Fetching all 57 in parallel made the
// page stutter (and the example animations stall) on slower computers.
const MAX_PARALLEL = 4;

// Must match CHUNK_SIZE in scripts/build_model.py
const CHUNK_SIZE = 10 * 1024 * 1024;

/**
 * Download (or read from the browser cache) every chunk and write it directly into a
 * single buffer of the model's final size. Each chunk is dropped as soon as it has been
 * copied, so the page never holds more than one full copy of the model. The old code
 * kept the chunks, a merged copy, and a Blob copy at the same time, which crashed tabs
 * on low-memory Chromebooks.
 */
export async function fetchAndMergeChunks(urls, version = 'default', onProgress, totalBytes) {
	CACHE_NAME = `${CACHE_PREFIX}-${version}`;
	await clearOldCaches();

	const model = new Uint8Array(totalBytes);
	const cache = await caches.open(CACHE_NAME).catch(() => null);
	let hasCache = false;
	let loaded = 0;

	const fetchOne = async (url, index) => {
		let buffer;
		const cached = cache ? await cache.match(url).catch(() => null) : null;
		if (cached) {
			hasCache = true;
			buffer = await cached.arrayBuffer();
		} else {
			const response = await fetch(url, { cache: 'no-store' });
			if (!response.ok) throw new Error(`Failed to fetch ${url}`);
			buffer = await response.arrayBuffer();
			// Saving is optional: some managed devices limit site storage
			if (cache) await cache.put(url, new Response(buffer)).catch(() => {});
		}
		model.set(new Uint8Array(buffer), index * CHUNK_SIZE);
		loaded += buffer.byteLength;
		onProgress?.(loaded);
	};

	let next = 0;
	const worker = async () => {
		while (next < urls.length) {
			const index = next++;
			await fetchOne(urls[index], index);
		}
	};
	await Promise.all(Array.from({ length: Math.min(MAX_PARALLEL, urls.length) }, worker));

	if (loaded !== totalBytes) throw new Error(`Model download incomplete: ${loaded} of ${totalBytes} bytes`);
	return { hasCache, model };
}

async function clearOldCaches() {
	const cacheNames = await caches.keys();
	await Promise.all(
		cacheNames.map((name) => {
			if (name !== CACHE_NAME && name.includes(CACHE_PREFIX)) {
				console.log(`Deleting old cache: ${name}`);
				return caches.delete(name);
			}
		})
	);
}
