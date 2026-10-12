import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	css: {
		preprocessorOptions: {
			scss: {
				additionalData: `@import 'src/styles/variables.scss';`,
				// Vite 6's Sass compiler resolves imports from these folders, not the working directory
				loadPaths: [process.cwd()],
				silenceDeprecations: ['import', 'global-builtin', 'legacy-js-api']
			}
		}
	},
	server: {
		fs: {
			// Allow serving files from one level up to the project root
			allow: ['..']
		}
	}
});
