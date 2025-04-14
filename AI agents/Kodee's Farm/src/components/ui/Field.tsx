import React, { useMemo } from 'react';
import { Box, Typography } from '@mui/material';

interface FieldProps {
  label: string;
  rows?: number;
  cols?: number;
  style?: React.CSSProperties;
  color?: string; // New prop for dynamic color
}

const Field: React.FC<FieldProps> = ({ label, rows = 5, cols = 10, style, color = '#999' }) => {
  // Debugging: Log the color prop to ensure it updates
  console.log(`Field "${label}" color:`, color);

  // Use useMemo to regenerate the grid whenever the color changes
  const grid = useMemo(
    () =>
      Array.from({ length: rows }, () =>
        Array.from({ length: cols }, () => color) // Use the color prop
      ),
    [rows, cols, color] // Regenerate grid when rows, cols, or color changes
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
          gap: '4px', // Reduced space between dots
        }}
      >
        {grid.map((row, rowIndex) =>
          row.map((dotColor, colIndex) => (
            <Box
              key={`${rowIndex}-${colIndex}`}
              sx={{
                width: '5px',
                height: '5px',
                backgroundColor: dotColor, // Use the dynamic color
                borderRadius: '100%', // Make it a dot
              }}
            />
          ))
        )}
      </Box>
    </Box>
  );
};

export default Field;