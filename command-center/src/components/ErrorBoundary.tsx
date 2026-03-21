import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Box, Typography, Button } from '@mui/material';

interface Props {
  children: ReactNode;
  fallbackLabel?: string;
}

interface State {
  hasError: boolean;
  error?: Error;
}

/**
 * Tab-level error boundary.
 * Catches any render/lifecycle errors in child subtrees and displays a
 * recovery UI instead of crashing the entire application.
 */
class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[ErrorBoundary]', error, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        <Box
          sx={{
            p: 4,
            textAlign: 'center',
            border: '1px solid #f44336',
            borderRadius: 2,
            m: 2,
          }}
        >
          <Typography variant="h6" gutterBottom sx={{ color: '#f44336' }}>
            {this.props.fallbackLabel ?? 'Something went wrong in this section'}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            {this.state.error?.message}
          </Typography>
          <Button
            variant="outlined"
            color="error"
            size="small"
            onClick={() => this.setState({ hasError: false, error: undefined })}
          >
            Retry
          </Button>
        </Box>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
