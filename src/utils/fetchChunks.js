const CACHE_PREFIX = 'onnx-model-cache';
// The cache name includes a version taken from the model's manifest, so browsers
// automatically drop old model files and re-download when the model changes.
let CACHE_NAME = `${CACHE_PREFIX}-default`;

// Downloads at most this many chunks at once. Fetching all 57 in parallel made the
// page stutter (and the example animations stall) on slower computers.
const MAX_PARALLEL = 4;

async function fetchModelChunks(chunkUrls, onProgress) {
	await clearOldCaches();

	let hasCache = false;
	let loaded = 0;
	const cache = await caches.open(CACHE_NAME);
	const buffers = new Array(chunkUrls.length);

	const fetchOne = async (url, index) => {
		const cached = await cache.match(url);
		let buffer;
		if (cached) {
			hasCache = true;
			buffer = await cached.arrayBuffer();
		} else {
			const response = await fetch(url, { cache: 'no-store' });
			if (!response.ok) throw new Error(`Failed to fetch ${url}`);
			buffer = await response.arrayBuffer();
			await cache.put(url, new Response(buffer));
		}
		buffers[index] = buffer;
		loaded += buffer.byteLength;
		onProgress?.(loaded);
	};

	let next = 0;
	const worker = async () => {
		while (next < chunkUrls.length) {
			const index = next++;
			await fetchOne(chunkUrls[index], index);
		}
	};
	await Promise.all(Array.from({ length: Math.min(MAX_PARALLEL, chunkUrls.length) }, worker));

	return { hasCache, modelBuffers: buffers };
}

export async function fetchAndMergeChunks(urls, version = 'default', onProgress) {
	CACHE_NAME = `${CACHE_PREFIX}-${version}`;
	const { hasCache, modelBuffers: chunks } = await fetchModelChunks(urls, onProgress);
	const totalSize = chunks.reduce((acc, chunk) => acc + chunk.byteLength, 0);
	const mergedArray = new Uint8Array(totalSize);
	let offset = 0;
	for (const chunk of chunks) {
		mergedArray.set(new Uint8Array(chunk), offset);
		offset += chunk.byteLength;
	}
	return { hasCache, mergedArray: mergedArray.buffer };
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
