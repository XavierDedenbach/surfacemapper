        # Adjust spacing based on complexity
        # Higher complexity = smaller spacing = more points
        adjusted_spacing = max_spacing - (max_spacing - min_spacing) * complexity
        adjusted_spacing = max(min_spacing, min(max_spacing, adjusted_spacing)) 