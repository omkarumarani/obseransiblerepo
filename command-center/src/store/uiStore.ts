import { create } from 'zustand';

type MainTab = 'live' | 'history' | 'scenarios' | 'learning';

interface UiState {
  currentTab: MainTab;
  lastRefreshAt: string;
  setCurrentTab: (tab: MainTab) => void;
  setLastRefreshAt: (isoTs: string) => void;
}

export const useUiStore = create<UiState>((set) => ({
  currentTab: 'live',
  lastRefreshAt: new Date().toISOString(),
  setCurrentTab: (tab) => set({ currentTab: tab }),
  setLastRefreshAt: (isoTs) => set({ lastRefreshAt: isoTs }),
}));
