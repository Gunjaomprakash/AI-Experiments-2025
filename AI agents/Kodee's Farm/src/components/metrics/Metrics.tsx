import React from "react";
import { Slider, Box, Typography } from "@mui/material";

interface Metric {
  name: string;
  value: number;
  color: string;
}

interface Field {
  name: string;
  metrics: Metric[];
}

interface MetricsProps {
  fields: Field[];
  onMetricChange: (index: number, newValue: number) => void;
}


const Metrics: React.FC<MetricsProps> = ({ fields, onMetricChange }) => {
  return (
    <Box>
      {fields.map((field, fieldIndex) => (
        <Box key={fieldIndex} mb={4}>
          <Typography variant="h6" gutterBottom>
            {field.name}
          </Typography>
          <Box>
            {field.metrics.map((metric, metricIndex) => (
              <Box key={metricIndex} mb={2}>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="body2" style={{ color: metric.color }}>
                    {metric.name}
                  </Typography>
                  <Typography variant="body2">{metric.value}%</Typography>
                </Box>
                <Slider
                  value={metric.value}
                  min={0}
                  max={100}
                  onChange={(_, newValue) =>
                    onMetricChange(metricIndex, newValue as number)
                  }
                  sx={{
                    color: metric.color,
                  }}
                />
              </Box>
            ))}
          </Box>
        </Box>
      ))}
    </Box>
  );
};

export default React.memo(Metrics);