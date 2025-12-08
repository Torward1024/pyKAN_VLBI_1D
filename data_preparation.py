import numpy as np
import pandas as pd

def calculate_baseline_amplitude(dataset, u_col='U', v_col='V', 
                                 re_col='Re', im_col='Im',
                                 baseline_col='Baseline', 
                                 amplitude_col='Amplitude',
                                 phase_col='Phase',
                                 inplace=False,
                                 sort_baselines=True):
    """
    Calculate Baseline and Amplitude columns and add them to the dataset.
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        Input dataset containing UV-coverage and complex visibility data
    u_col : str, default='U'
        Column name for U coordinates
    v_col : str, default='V'
        Column name for V coordinates
    re_col : str, default='Re'
        Column name for real part of visibility
    im_col : str, default='Im'
        Column name for imaginary part of visibility
    baseline_col : str, default='Baseline'
        Column name for calculated baseline projection
    amplitude_col : str, default='Amplitude'
        Column name for calculated amplitude
    phase_col: str, default='Phase'
        Column name for calculated phase
    inplace : bool, default=False
        If True, modify the original dataset. If False, return a copy.
    
    Returns:
    --------
    pd.DataFrame
        Dataset with added Baseline and Amplitude columns (if inplace=False)
    None
        If inplace=True, modifies the original dataset and returns None
    
    Notes:
    ------
    Baseline = sqrt(U^2 + V^2)
    Amplitude = sqrt(Re^2 + Im^2)
    Phase = atan(Im/Re)
    """
    # work with copy if not inplace
    if not inplace:
        dataset_v = dataset.copy()
    else:
        dataset_v = dataset
    
    # сalculate baseline projection
    dataset_v[baseline_col] = np.sqrt(dataset_v[u_col]**2 + dataset_v[v_col]**2)
    
    # сalculate amplitude
    dataset_v[amplitude_col] = np.sqrt(dataset_v[re_col]**2 + dataset_v[im_col]**2)

    # сalculate amplitude
    dataset_v[phase_col] = np.arctan2(dataset_v[im_col], dataset_v[re_col])
    
    if sort_baselines:
        dataset_v.sort_values('Baseline').reset_index(drop=True)
    
    if not inplace:
        return dataset_v

def calculate_baseline_amplitude_angle(dataset, u_col='U', v_col='V', 
                                 re_col='Re', im_col='Im',
                                 baseline_col='Baseline', 
                                 angle_col='Angle',
                                 amplitude_col='Amplitude',
                                 phase_col='Phase',
                                 inplace=False,
                                 sort_baselines=True):
    """
    Calculate Baseline, Angle, and Amplitude columns and add them to the dataset.
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        Input dataset containing UV-coverage and complex visibility data
    u_col : str, default='U'
        Column name for U coordinates
    v_col : str, default='V'
        Column name for V coordinates
    re_col : str, default='Re'
        Column name for real part of visibility
    im_col : str, default='Im'
        Column name for imaginary part of visibility
    baseline_col : str, default='Baseline'
        Column name for calculated baseline projection
    amplitude_col : str, default='Amplitude'
        Column name for calculated amplitude
    phase_col: str, default='Phase'
        Column name for calculated phase
    inplace : bool, default=False
        If True, modify the original dataset. If False, return a copy.
    
    Returns:
    --------
    pd.DataFrame
        Dataset with added Baseline and Amplitude columns (if inplace=False)
    None
        If inplace=True, modifies the original dataset and returns None
    
    Notes:
    ------
    Baseline = sqrt(U^2 + V^2)
    Amplitude = sqrt(Re^2 + Im^2)
    Phase = atan(Im/Re)
    """
    # work with copy if not inplace
    if not inplace:
        dataset_v = dataset.copy()
    else:
        dataset_v = dataset
    
    # сalculate baseline projection
    dataset_v[baseline_col] = np.sqrt(dataset_v[u_col]**2 + dataset_v[v_col]**2)

    # сalculate angle
    dataset_v[angle_col] = np.arctan2(dataset_v[v_col], dataset_v[u_col])
    
    # сalculate amplitude
    dataset_v[amplitude_col] = np.sqrt(dataset_v[re_col]**2 + dataset_v[im_col]**2)

    # сalculate amplitude
    dataset_v[phase_col] = np.arctan2(dataset_v[im_col], dataset_v[re_col])
    
    if sort_baselines:
        dataset_v.sort_values('Baseline').reset_index(drop=True)
    
    if not inplace:
        return dataset_v

def cross_section(dataset, delta, alpha_deg, length=None):
    """
    Extract a cross-section from the dataset based on distance from a line.
    
    This function selects data points that lie within a specified distance (delta)
    from a line defined by a given angle (alpha_deg). Optionally, can also
    constrain the projection length along the line.
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        Input dataset containing 'U' and 'V' columns
    delta : float
        Maximum distance from the line for point selection
    alpha_deg : float
        Angle of the line in degrees (0 = horizontal, 90 = vertical)
    length : float, optional
        Maximum absolute projection length along the line.
        If None, no length constraint is applied.
    
    Returns:
    --------
    filtered_data : pd.DataFrame
        Subset of the dataset containing only selected points
    mask : pd.Series of bool
        Boolean mask indicating which points were selected
    distance : pd.Series
        Distance of each point from the line
    
    Notes:
    ------
    The line is defined by: distance = |U*sin(alpha) - V*cos(alpha)| <= delta
    Projection along the line: projection = U*cos(alpha) + V*sin(alpha)
    
    Examples:
    ---------
    >>> filtered, mask, distances = cross_section(dataset, delta=0.5, alpha_deg=45)
    >>> filtered_len, mask_len, distances_len = cross_section(dataset, delta=0.5, 
    ...                                                        alpha_deg=45, length=10)
    """
    # convert angle to radians
    alpha = np.radians(alpha_deg)   
    
    # calculate distance from the line
    distance = np.abs(dataset['U'] * np.sin(alpha) - dataset['V'] * np.cos(alpha))
    mask_distance = distance <= delta
    
    # apply length constraint if specified
    if length is not None:
        projection = dataset['U'] * np.cos(alpha) + dataset['V'] * np.sin(alpha)
        mask_length = np.abs(projection) <= length
        mask = mask_distance & mask_length
    else:
        mask = mask_distance
    
    # create filtered dataset
    filtered_data = dataset[mask].copy()
    
    # print summary statistics
    print(f"Angle: {alpha_deg}, (deg.)")
    print(f"Cross-section width: {delta}")
    print(f"Number of points: {mask.sum()}/{len(dataset)} ({mask.sum()/len(dataset)*100:.1f}%)")
    
    return filtered_data, mask, distance

def add_noise(df, re_noise_percent=5, im_noise_percent=5, random_seed=42):
    """
    Add Gaussian noise to the real and imaginary parts of visibility data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset containing 'Re' and 'Im' columns
    re_noise_percent : float, default=5
        Noise level for real part as percentage of standard deviation
    im_noise_percent : float, default=5
        Noise level for imaginary part as percentage of standard deviation
    random_seed : int, default=42
        Seed for random number generator for reproducible results
    
    Returns:
    --------
    df_noisy : pd.DataFrame
        Copy of the input dataset with added noise to 'Re' and 'Im' columns
    
    Notes:
    ------
    Noise is calculated as: N(0, sigma * noise_percent/100)
    where sigma is the standard deviation of the respective column.
    Other columns in the dataframe remain unchanged.
    
    Examples:
    ---------
    >>> df_noisy = add_noise(df, re_noise_percent=5, im_noise_percent=5)
    >>> df_noisy_low = add_noise(df, re_noise_percent=1, im_noise_percent=1, random_seed=123)
    """
    # initialize random number generator with seed
    rng = np.random.RandomState(random_seed)
    
    # create a copy to avoid modifying original data
    df_noisy = df.copy()
    
    # calculate standard deviations
    re_std = df['Re'].std()
    im_std = df['Im'].std()
    
    # calculate noise standard deviations
    re_noise_std = re_std * (re_noise_percent / 100)
    im_noise_std = im_std * (im_noise_percent / 100)
    
    # add noise to real and imaginary parts
    df_noisy['Re'] = df['Re'] + rng.normal(0, re_noise_std, len(df))
    df_noisy['Im'] = df['Im'] + rng.normal(0, im_noise_std, len(df))
    
    return df_noisy
    
def select_by_baseline_ranges(dataset, baseline_col='Baseline', 
                                ranges=None, num_ranges=3, 
                                range_size_bounds=(0.01, 0.1),
                                random_seed=None, mode='specified',
                                min_gap=0.0, ensure_non_overlapping=True,
                                verbose=True):
    """
    Enhanced version with more control over range generation.
    
    Parameters:
    -----------
    min_gap : float, default=0.0
        Minimum gap between ranges (for random mode)
    ensure_non_overlapping : bool, default=True
        Ensure random ranges don't overlap (for random mode)
    verbose : bool, default=True
        Whether to print summary information
    
    Returns:
    --------
    selected_data : pd.DataFrame
    rest_data : pd.DataFrame
    selected_ranges : list of tuples
    """
    # validate inputs
    if mode not in ['specified', 'random', 'all']:
        raise ValueError(f"Invalid mode: {mode}")
    
    if baseline_col not in dataset.columns:
        raise KeyError(f"Column '{baseline_col}' not found")
    
    # Initialize
    baseline_mask = pd.Series(False, index=dataset.index)
    selected_ranges = []
    
    if mode == 'all':
        # select all data
        baseline_mask = pd.Series(True, index=dataset.index)
        selected_ranges = [(dataset[baseline_col].min(), dataset[baseline_col].max())]
    
    elif mode == 'random':
        # validate parameters
        if num_ranges <= 0:
            raise ValueError("num_ranges must be positive")
        
        # get baseline statistics
        baseline_min = dataset[baseline_col].min()
        baseline_max = dataset[baseline_col].max()
        baseline_span = baseline_max - baseline_min
        
        # validate range size
        min_size, max_size = range_size_bounds
        if min_size <= 0 or max_size <= 0:
            raise ValueError("Range sizes must be positive")
        if min_size > max_size:
            min_size, max_size = max_size, min_size  # Auto-correct
        
        # calculate maximum possible ranges
        max_possible_ranges = int(baseline_span / (min_size + min_gap))
        if max_possible_ranges < num_ranges and ensure_non_overlapping:
            if verbose:
                print(f"Warning: Can only fit {max_possible_ranges} non-overlapping ranges")
            num_ranges = min(num_ranges, max_possible_ranges)
        
        # initialize random generator
        rng = np.random.RandomState(random_seed)
        
        # generate ranges
        generated_ranges = []
        attempts = 0
        max_attempts = num_ranges * 10  # prevent infinite loop
        
        while len(generated_ranges) < num_ranges and attempts < max_attempts:
            attempts += 1
            
            # generate candidate range
            range_size = rng.uniform(min_size, max_size)
            start = rng.uniform(baseline_min, baseline_max - range_size)
            end = start + range_size
            end = min(end, baseline_max)  # Ensure within bounds
            
            # check for overlaps if required
            if ensure_non_overlapping:
                overlapping = False
                for existing_start, existing_end in generated_ranges:
                    if not (end + min_gap <= existing_start or 
                           start >= existing_end + min_gap):
                        overlapping = True
                        break
                
                if overlapping:
                    continue
            
            generated_ranges.append((start, end))
        
        # sort ranges by start value
        generated_ranges.sort(key=lambda x: x[0])
        selected_ranges = generated_ranges
        
        # apply masks
        for start, end in selected_ranges:
            mask = dataset[baseline_col].between(start, end)
            baseline_mask = baseline_mask | mask
    
    elif mode == 'specified':
        # process specified ranges
        if ranges is None:
            raise ValueError("ranges parameter required for 'specified' mode")
        
        # handle single range
        if isinstance(ranges, tuple):
            ranges = [ranges]
        
        # validate and apply each range
        for i, (start, end) in enumerate(ranges):
            if start >= end:
                raise ValueError(f"Range {i}: start must be less than end")
            
            selected_ranges.append((start, end))
            mask = dataset[baseline_col].between(start, end)
            baseline_mask = baseline_mask | mask
    
    # create output datasets
    selected_data = dataset[baseline_mask].copy()
    rest_data = dataset[~baseline_mask].copy()
    
    # summary output
    if verbose:
        total_points = len(dataset)
        selected_points = len(selected_data)
        percentage = selected_points / total_points * 100 if total_points > 0 else 0
        
        print(f"Mode: {mode}")
        print(f"Selected baseline ranges: {selected_ranges}")
        print(f"Points selected: {selected_points:,} from {total_points:,} "
              f"({percentage:.1f}%)")
        print(f"Selected range sizes: {[(end-start) for start, end in selected_ranges]}")
    
    return selected_data, rest_data, selected_ranges
