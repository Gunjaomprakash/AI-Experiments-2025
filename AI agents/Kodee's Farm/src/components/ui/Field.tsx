import React from 'react';
import { Box, Typography } from '@mui/material';

interface FieldProps {
  label: string;
  rows?: number;
  cols?: number;
  style?: React.CSSProperties;
}

const Field: React.FC<FieldProps> = ({ label, rows = 5, cols = 10, style }) => {
  // Create a 2D grid with default color for the dots (dotted pattern)
  const grid = Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => '#999') // Default dot color (gray)
  );

  return (
    <Box style={style}>
      {/* Field Label */}
      <Typography variant="h6" gutterBottom>
        {label}
      </Typography>
      {/* Render the grid */}
      <Box
        sx={{
          display: 'grid',
          gridTemplateRows: `repeat(${rows}, 1fr)`,
          gridTemplateColumns: `repeat(${cols}, 1fr)`,
          gap: '5px', // Reduced space between dots
        }}
      >
        {grid.map((row, rowIndex) =>
          row.map((color, colIndex) => (
            <Box
              key={`${rowIndex}-${colIndex}`}
              sx={{
                width: '3px',
                height: '3px',
                backgroundColor:'#30792e',
                borderRadius: '50%', // Make it a dot
              }}
            />
          ))
        )}
      </Box>
    </Box>
  );
};

export default Field;