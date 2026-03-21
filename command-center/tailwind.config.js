/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{js,jsx,ts,tsx}', './index.html'],
  theme: {
    extend: {
      colors: {
        aiops: {
          deep: '#0a0e27',
          panel: '#1a1f3a',
          accent: '#2196f3',
        },
      },
    },
  },
  plugins: [],
};
