import { writable, derived, readable } from 'svelte/store';
import * as ort from 'onnxruntime-web';
import tailwindConfig from '../../tailwind.config';
import resolveConfig from 'tailwindcss/resolveConfig';
import { ex0 } from '~/constants/examples';
import { textPages } from '~/utils/textbookPages';
import { MODEL_NAME } from '~/constants/model';

const { theme } = resolveConfig(tailwindConfig);

export const attentionHeadIdxTemp = writable(0);
export const attentionHeadIdx = writable(0);
export const blockIdxTemp = writable(0);
export const blockIdx = writable(0);
export const isOnBlockTransition = writable(false);

export const isOnAnimation = writable(false);

// Textbook state management
export const textbookCurrentPage = writable<number>(0);
export const textbookPreviousPage = writable<number>(-1);
export const textbookCurrentPageId = writable<string>(textPages[0].id);
export const textbookPreviousPageId = writable<string>('');
export const isTextbookOpen = writable<boolean>(true);

// is transformer running?
export const isModelRunning = writable(false);
export const isFetchingModel = writable(true);
export const isLoaded = writable(false);

// Model download progress, shown as a bar under the prompt box
export const downloadProgress = writable<{
	loaded: number;
	total: number;
	phase: 'idle' | 'downloading' | 'preparing' | 'done';
}>({ loaded: 0, total: 0, phase: 'idle' });

export const inputTextExample = [
	'Data visualization empowers users to',
	'Artificial Intelligence is transforming the',
	'As the spaceship was approaching the',
	'On the deserted planet they discovered a',
	'IEEE VIS conference highlights the'
];

const initialExIdx = 0;
export const selectedExampleIdx = writable<number>(initialExIdx);

export const modelSession = writable<ort.InferenceSession>();

// transformer model output
export const modelData = writable<ModelData>(ex0);
export const predictedToken = writable<Probability>();
export const tokens = writable<string[]>(ex0?.tokens);
export const tokenIds = writable<number[]>(ex0?.tokenIds);

export const modelMetaMap: Record<string, ModelMetaData> = {
	gpt2: { layer_num: 12, attention_head_num: 12, dimension: 768, params: '124 million', label: 'GPT-2 (small)' },
	'gpt2-medium': { layer_num: 24, attention_head_num: 16, dimension: 1024, params: '355 million', label: 'GPT-2 (medium)' },
	'gpt2-large': { layer_num: 36, attention_head_num: 20, dimension: 1280, params: '774 million', label: 'GPT-2 (large)' }
};
export const activeModelMeta = modelMetaMap[MODEL_NAME];

// selected token vector
export const highlightedToken = writable<HighlightedToken>({
	index: null,
	value: null,
	fix: false
});

export const highlightedHead = writable<HighlightedToken>({
	index: null,
	value: null,
	fix: false
});

// expanded block
export const expandedBlock = writable<ExpandedBlock>({ id: null });
export const isExpandOrCollapseRunning = writable(false);

// user input text
export const inputText = writable(inputTextExample[initialExIdx]);
// export const tokens = derived(inputText, ($inputText) => $inputText.trim().split(' '));

// selected model and meta data
const initialSelectedModel = MODEL_NAME;
export const selectedModel = writable(initialSelectedModel);
export const modelMeta = derived(selectedModel, ($selectedModel) => modelMetaMap[$selectedModel]);

// Temperature setting
export const initialTemperature = 0.8;
export const temperature = writable(initialTemperature);

// Sampling
export const sampling = writable<Sampling>({ type: 'top-k', value: 5 });

// Prediction visual
export const highlightedIndex = writable(null);
export const finalTokenIndex = writable(null);

// Visual element style
export const rootRem = 16;
export const minVectorHeight = 12;
export const maxVectorHeight = 30;
export const maxVectorScale = 3.4;

export const vectorHeight = writable(0);
export const headContentHeight = writable(0);
export const headGap = { x: 5, y: 8, scale: 0 };

export const isBoundingBoxActive = writable(false);

export const predictedColor = theme.colors.purple[600];

// Interactivity
export const hoveredPath = writable();
export const hoveredMatrixCell = writable({ row: null, col: null });
export const weightPopover = writable();
export const tooltip = writable();

export const isMobile = readable(false, (set) => {
	if (typeof window !== 'undefined') {
		// Only run in browser environment
		const userAgent = navigator.userAgent.toLowerCase();
		set(/android|iphone|ipad|ipod/i.test(userAgent));
	}
	return () => {}; // Cleanup function
});

// User identification
export const userId = writable<string | null>(null);