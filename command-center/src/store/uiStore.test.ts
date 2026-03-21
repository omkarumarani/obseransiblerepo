import { describe, it, expect, beforeEach } from 'vitest';
import { useUiStore } from '../uiStore';

describe('uiStore', () => {
  beforeEach(() => {
    // Reset to default state before each test
    useUiStore.setState({ currentTab: 'live', lastRefreshAt: new Date().toISOString() });
  });

  it('defaults to live tab', () => {
    expect(useUiStore.getState().currentTab).toBe('live');
  });

  it('setCurrentTab switches to history', () => {
    useUiStore.getState().setCurrentTab('history');
    expect(useUiStore.getState().currentTab).toBe('history');
  });

  it('setCurrentTab switches to scenarios', () => {
    useUiStore.getState().setCurrentTab('scenarios');
    expect(useUiStore.getState().currentTab).toBe('scenarios');
  });

  it('setLastRefreshAt stores the given ISO string', () => {
    const iso = '2026-03-21T12:00:00.000Z';
    useUiStore.getState().setLastRefreshAt(iso);
    expect(useUiStore.getState().lastRefreshAt).toBe(iso);
  });

  it('lastRefreshAt is a valid ISO date string by default', () => {
    const ts = useUiStore.getState().lastRefreshAt;
    expect(new Date(ts).toISOString()).toBe(ts);
  });
});
