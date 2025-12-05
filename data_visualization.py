import matplotlib.pyplot as plt
import numpy as np

def plot_scatter(x, y, alpha=0.1, size=5, color='blue', 
                 xlabel=None, ylabel=None, title=None, 
                 grid=True, figsize=(13, 5)):
    """
    Create a scatter plot with customizable parameters.
    
    Parameters:
    -----------
    x : pd.Series or array-like
        Data for x-axis
    y : pd.Series or array-like
        Data for y-axis
    alpha : float, default=0.1
        Transparency of points
    size : int or float, default=5
        Size of points
    color : str, default='blue'
        Color of points
    xlabel : str, optional
        Label for x-axis
    ylabel : str, optional
        Label for y-axis
    title : str, optional
        Title of the plot
    grid : bool, default=True
        Whether to show grid
    figsize : tuple, default=(13, 5)
        Figure size (width, height) in inches
    
    Returns:
    --------
    None (displays the plot)
    """
    plt.figure(figsize=figsize)
    plt.scatter(x, y, 
                alpha=alpha, 
                s=size, 
                c=color)
    
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    
    if grid:
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_uv_coverage(dataset, alpha_test, delta_test, 
                     amplitude_col='Amplitude', u_col='U', v_col='V',
                     figsize=(13, 6), point_size=3, alpha_val=0.6, 
                     cmap='viridis', selected_color='red'):
    """
    Plot UV-coverage dataset with selection based on cross-section criteria.
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        Input dataset containing UV-coverage and amplitude data
    alpha_test : float
        Rotation angle in degrees for cross-section line
    delta_test : float
        Width parameter for selection (distance to line)
    amplitude_col : str, default='Amplitude'
        Column name for amplitude values
    u_col : str, default='U'
        Column name for U coordinates
    v_col : str, default='V'
        Column name for V coordinates
    figsize : tuple, default=(13, 6)
        Figure size (width, height) in inches
    point_size : int, default=3
        Size of scatter plot points
    alpha_val : float, default=0.6
        Transparency of scatter plot points
    cmap : str, default='viridis'
        Colormap for amplitude visualization
    selected_color : str, default='red'
        Color for selected points
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    axs : np.ndarray of matplotlib.axes.Axes
        Array of axes objects
    """
    # calculate projection and selection mask
    alpha_rad = np.radians(alpha_test)
    projection = dataset[u_col] * np.cos(alpha_rad) + dataset[v_col] * np.sin(alpha_rad)
    distance_to_line = np.abs(dataset[u_col] * np.sin(alpha_rad) - dataset[v_col] * np.cos(alpha_rad))
    mask = distance_to_line <= delta_test
    
    # create figure and axes
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    # common scatter plot parameters
    scatter_kwargs = dict(s=point_size, alpha=alpha_val, cmap=cmap, edgecolors='none')
    
    # initial dataset
    sc1 = axs[0].scatter(dataset[u_col], dataset[v_col], 
                         c=dataset[amplitude_col], **scatter_kwargs)
    axs[0].set_xlabel(f'{u_col}, (Earth Diameters)')
    axs[0].set_ylabel(f'{v_col}, (Earth Diameters)')
    axs[0].set_title('Initial Dataset (u,v-coverage)')
    axs[0].set_aspect('equal')
    axs[0].grid(True, alpha=0.3)
    plt.colorbar(sc1, ax=axs[0], label='Visibility Amplitude', 
                 fraction=0.046, pad=0.04)
    
    # dataset with cross-section
    sc2 = axs[1].scatter(dataset[u_col], dataset[v_col], 
                         c=dataset[amplitude_col], **scatter_kwargs)
    axs[1].set_xlabel(f'{u_col}, (Earth Diameters)')
    axs[1].set_ylabel(f'{v_col}, (Earth Diameters)')
    axs[1].set_title(f'Cross-section Dataset: alpha = {alpha_test}, delta = {delta_test}')
    axs[1].set_aspect('equal')
    axs[1].grid(True, alpha=0.3)
    
    # add rectangle for cross-section visualization
    cos_a, sin_a = np.cos(alpha_rad), np.sin(alpha_rad)
    length = max(np.abs(projection).max(), 0.5) * 1.2
    width_vec = np.array([-sin_a, cos_a]) * delta_test
    length_vec = np.array([cos_a, sin_a]) * length
    corners = [
        -length_vec/2 - width_vec,
        -length_vec/2 + width_vec,
        +length_vec/2 + width_vec,
        +length_vec/2 - width_vec
    ]
    rect = plt.Polygon(corners, facecolor='none', edgecolor='red', 
                       linewidth=2, linestyle='--')
    axs[1].add_patch(rect)
    
    # highlight selected points
    axs[1].scatter(dataset[u_col][mask], dataset[v_col][mask], 
                   c=selected_color, s=point_size*2, alpha=0.8, 
                   label='Selected')
    axs[1].legend()
    
    # add colorbar to second plot
    plt.colorbar(sc2, ax=axs[1], label='Visibility Amplitude', 
                 fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    
    return fig, axs

def plot_visibility_components(filtered_data, baseline_col='Baseline', 
                              amplitude_col='Amplitude', real_col='Re', 
                              imag_col='Im', alpha=0.1, size=5,
                              amplitude_color='blue', real_color='red', 
                              imag_color='yellow', figsize=(13, 5),
                              xlabel='Baseline Projection, (ED)', 
                              ylabel='Visibility Amplitude',
                              title_prefix='Cross-section Dataset',
                              alpha_test=None, delta_test=None,
                              grid=True):
    """
    Plot visibility amplitude, real, and imaginary components vs baseline projection.
    
    Parameters:
    -----------
    filtered_data : pd.DataFrame
        Dataset containing baseline projection and visibility components
    baseline_col : str, default='Baseline'
        Column name for baseline projection values
    amplitude_col : str, default='Amplitude'
        Column name for amplitude values
    real_col : str, default='Re'
        Column name for real part values
    imag_col : str, default='Im'
        Column name for imaginary part values
    alpha : float, default=0.1
        Transparency of points
    size : int, default=5
        Size of points
    amplitude_color : str, default='blue'
        Color for amplitude points
    real_color : str, default='red'
        Color for real part points
    imag_color : str, default='yellow'
        Color for imaginary part points
    figsize : tuple, default=(13, 5)
        Figure size (width, height) in inches
    xlabel : str, default='Baseline Projection, (ED)'
        Label for x-axis
    ylabel : str, default='Visibility Amplitude'
        Label for y-axis
    title_prefix : str, default='Cross-section Dataset'
        Prefix for plot title
    alpha_test : float, optional
        Rotation angle parameter for title
    delta_test : float, optional
        Width parameter for title
    grid : bool, default=True
        Whether to show grid
    
    Returns:
    --------
    None (displays the plot)
    """
    plt.figure(figsize=figsize)
    
    # plot amplitude component
    plt.scatter(filtered_data[baseline_col], 
                filtered_data[amplitude_col], 
                alpha=alpha, 
                s=size,
                c=amplitude_color,
                label='Amplitude')
    
    # plot real component
    plt.scatter(filtered_data[baseline_col], 
                filtered_data[real_col], 
                alpha=alpha, 
                s=size,
                c=real_color,
                label='Real part')
    
    # plot imaginary component
    plt.scatter(filtered_data[baseline_col], 
                filtered_data[imag_col], 
                alpha=alpha, 
                s=size,
                c=imag_color,
                label='Imaginary part')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # create title
    title_parts = [title_prefix]
    if alpha_test is not None and delta_test is not None:
        title_parts.append(f'alpha = {alpha_test}, delta = {delta_test}')
    title_parts.append('Visibility Amplitude vs. Baseline Projection')
    
    plt.title('\n'.join(title_parts))
    
    if grid:
        plt.grid(True, alpha=0.3)
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_amplitude_phase_with_selection(filtered_data, selected_data, ranges,
                                        baseline_col='Baseline', amplitude_col='Amplitude',
                                        phase_col='Phase',
                                        re_col='Re', im_col='Im', alpha_test=None, delta_test=None,
                                        figsize=(16, 6), alpha_full=0.65, alpha_selected=0.7,
                                        full_color='lightblue', selected_color='red',
                                        edgecolor='darkred', linewidth=0.5):
    """
    Plot amplitude and phase of visibility data with highlighted selected ranges.
    
    Parameters:
    -----------
    filtered_data : pd.DataFrame
        Full cross-section dataset containing all points
    selected_data : pd.DataFrame
        Dataset containing selected points within specified ranges
    ranges : list of tuples
        Baseline ranges used for selection [(start1, end1), (start2, end2), ...]
    baseline_col : str, default='Baseline'
        Column name for baseline projection values
    amplitude_col : str, default='Amplitude'
        Column name for amplitude values
    phase_col: str, default='Phase'
        Column name for phase values
    re_col : str, default='Re'
        Column name for real part values
    im_col : str, default='Im'
        Column name for imaginary part values
    alpha_test : float, optional
        Rotation angle parameter for title
    delta_test : float, optional
        Width parameter for title
    figsize : tuple, default=(16, 6)
        Figure size (width, height) in inches
    alpha_full : float, default=0.65
        Transparency for full dataset points
    alpha_selected : float, default=0.7
        Transparency for selected dataset points
    full_color : str, default='lightblue'
        Color for full dataset points
    selected_color : str, default='red'
        Color for selected dataset points
    edgecolor : str, default='darkred'
        Edge color for selected points
    linewidth : float, default=0.5
        Line width for selected points edges
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object containing both subplots
    axs : numpy.ndarray of matplotlib.axes.Axes
        Array of axes objects [amplitude_ax, phase_ax]
    """ 
    # create figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    # amplitude
    # full dataset
    axs[0].scatter(filtered_data[baseline_col],
                   filtered_data[amplitude_col],
                   alpha=alpha_full,
                   s=8,
                   c=full_color,
                   label=f'Cross-section ({len(filtered_data):,} points)')
    
    # selected data
    axs[0].scatter(selected_data[baseline_col],
                   selected_data[amplitude_col],
                   alpha=alpha_selected,
                   s=12,
                   c=selected_color,
                   edgecolors=edgecolor,
                   linewidth=linewidth,
                   label=f'Selected: {len(selected_data):,} points in {len(ranges)} ranges')
    
    # add vertical lines for ranges
    for i, (start, stop) in enumerate(ranges):
        axs[0].axvline(start, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        axs[0].axvline(stop, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # configure amplitude plot
    axs[0].set_xlabel(f'{baseline_col}, (Earth Diameters)', fontsize=12)
    axs[0].set_ylabel('Visibility Amplitude', fontsize=12)
    
    # create title for amplitude plot
    title_parts = []
    if alpha_test is not None and delta_test is not None:
        title_parts.append(f'Cross-section: alpha = {alpha_test}, delta = {delta_test}')
    title_parts.append(f'Selected Ranges: {ranges}')
    axs[0].set_title('Amplitude vs Baseline Projection\n' + '\n'.join(title_parts), 
                     fontsize=13, pad=15)
    
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=11, loc='upper right')
    
    # phase
    # full dataset phase
    axs[1].scatter(filtered_data[baseline_col],
                   filtered_data['Phase'],
                   alpha=alpha_full,
                   s=8,
                   c=full_color,
                   label=f'Cross-section ({len(filtered_data):,} points)')
    
    # plot selected data phase
    axs[1].scatter(selected_data[baseline_col],
                   selected_data['Phase'],
                   alpha=alpha_selected,
                   s=12,
                   c=selected_color,
                   edgecolors=edgecolor,
                   linewidth=linewidth,
                   label=f'Selected: {len(selected_data):,} points in {len(ranges)} ranges')
    
    # add vertical lines for ranges
    for i, (start, stop) in enumerate(ranges):
        axs[1].axvline(start, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        axs[1].axvline(stop, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # configure phase plot
    axs[1].set_xlabel(f'{baseline_col}, (Earth Diameters)', fontsize=12)
    axs[1].set_ylabel('Phase (radians)', fontsize=12)
    axs[1].set_title('Phase vs Baseline Projection\n' + '\n'.join(title_parts), 
                     fontsize=13, pad=15)
    
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=11, loc='upper right')
    
    # adjust layout and show
    plt.tight_layout()
    plt.show()
    
    return fig, axs

def plot_predictions_comparison(filtered_data, selected_data, 
                                results_amplitude, results_phase,
                                used_ranges,
                                plot_amplitude=True, plot_phase=True,
                                baseline_col='Baseline', amplitude_col='Amplitude',
                                phase_col='Phase', alpha_test=None, delta_test=None,
                                figsize=None, subplot_kwargs=None):
    """
    Flexible function to plot amplitude and/or phase with predictions.
    
    Parameters:
    -----------
    filtered_data : pd.DataFrame
        Full cross-section dataset
    selected_data : pd.DataFrame
        Selected points dataset
    results_amplitude : array-like
        Model predictions for amplitude
    results_phase : array-like
        Model predictions for phase
    used_ranges : list of tuples
        Baseline ranges used for selection [(start1, end1), ...]
    plot_amplitude : bool, default=True
        Whether to plot amplitude comparison
    plot_phase : bool, default=True
        Whether to plot phase comparison
    baseline_col : str, default='Baseline'
        Column name for baseline projection
    amplitude_col : str, default='Amplitude'
        Column name for amplitude values
    phase_col : str, default='Phase'
        Column name for phase values
    alpha_test : float, optional
        Rotation angle parameter for title
    delta_test : float, optional
        Width parameter for title
    figsize : tuple, optional
        Figure size, auto-calculated if None
    subplot_kwargs : dict, optional
        Additional arguments for plt.subplots
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
    axs : single Axes or array of Axes
    
    Notes:
    ------
    Both results_amplitude and results_phase should have same length as selected_data
    """
    # validate inputs
    if len(results_amplitude) != len(selected_data):
        raise ValueError(f"results_amplitude length ({len(results_amplitude)}) "
                         f"must match selected_data length ({len(selected_data)})")
    
    if len(results_phase) != len(selected_data):
        raise ValueError(f"results_phase length ({len(results_phase)}) "
                         f"must match selected_data length ({len(selected_data)})")
    
    # determine number of subplots
    n_plots = plot_amplitude + plot_phase
    if n_plots == 0:
        raise ValueError("At least one of plot_amplitude or plot_phase must be True")
    
    # auto-calculate figsize if not provided
    if figsize is None:
        figsize = (6 * n_plots + 4, 6)
    
    # create subplots
    if subplot_kwargs is None:
        subplot_kwargs = {}
    
    if n_plots > 1:
        fig, axs = plt.subplots(1, n_plots, figsize=figsize, **subplot_kwargs)
        axs = axs.flatten()
    else:
        fig, ax = plt.subplots(figsize=figsize, **subplot_kwargs)
        axs = [ax]
    
    # create common title parts
    title_parts = []
    if alpha_test is not None and delta_test is not None:
        title_parts.append(f'Cross-section: alpha = {alpha_test}, delta = {delta_test}')
    title_parts.append(f'Selected Ranges: {used_ranges}')
    
    ax_index = 0
    
    # amplitude
    if plot_amplitude:
        ax = axs[ax_index]
        ax_index += 1
        
        # plot data
        ax.scatter(filtered_data[baseline_col], filtered_data[amplitude_col],
                   alpha=0.65, s=8, c='lightblue',
                   label=f'Cross-section ({len(filtered_data):,} points)')
        
        ax.scatter(selected_data[baseline_col], selected_data[amplitude_col],
                   alpha=0.7, s=12, c='red', edgecolors='darkred', linewidth=0.5,
                   label=f'Selected data: {len(selected_data):,} points')
        
        ax.scatter(selected_data[baseline_col], results_amplitude,
                   alpha=0.7, s=25, c='yellow', edgecolors='orange',
                   linewidth=1.0, marker='s',
                   label=f'KAN predictions ({len(selected_data):,} points)')
        
        # add range markers
        for start, stop in used_ranges:
            ax.axvline(start, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.axvline(stop, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # configure
        ax.set_xlabel(f'{baseline_col}, (Earth Diameters)', fontsize=12)
        ax.set_ylabel('Visibility Amplitude', fontsize=12)
        ax.set_title('Amplitude vs Baseline Projection\n' + '\n'.join(title_parts), 
                     fontsize=13, pad=15)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='upper right')
    
    # phase
    if plot_phase:
        ax = axs[ax_index] if n_plots > 1 else axs[0]
        
        # plot data
        ax.scatter(filtered_data[baseline_col], filtered_data[phase_col],
                   alpha=0.65, s=8, c='lightblue',
                   label=f'Cross-section ({len(filtered_data):,} points)')
        
        ax.scatter(selected_data[baseline_col], selected_data[phase_col],
                   alpha=0.7, s=12, c='red', edgecolors='darkred', linewidth=0.5,
                   label=f'Selected data: {len(selected_data):,} points')
        
        ax.scatter(selected_data[baseline_col], results_phase,
                   alpha=0.7, s=25, c='yellow', edgecolors='orange',
                   linewidth=1.0, marker='s',
                   label=f'KAN predictions ({len(selected_data):,} points)')
        
        # add range markers
        for start, stop in used_ranges:
            ax.axvline(start, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.axvline(stop, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # configure
        ax.set_xlabel(f'{baseline_col}, (Earth Diameters)', fontsize=12)
        ax.set_ylabel('Visibility Phase (radians)', fontsize=12)
        ax.set_title('Phase vs Baseline Projection\n' + '\n'.join(title_parts), 
                     fontsize=13, pad=15)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    return fig, axs

def plot_whole_predictions(filtered_data, results_amplitude, results_phase=None,
                                    used_ranges=None, plot_amplitude=True, plot_phase=True,
                                    baseline_col='Baseline', amplitude_col='Amplitude',
                                    phase_col='Phase', title_prefix='Total Selected Cross-section',
                                    alpha_test=None, delta_test=None, figsize=None):
    """
    Flexible function to plot predictions for entire dataset.
    
    Parameters:
    -----------
    title_prefix : str, default='Total Selected Cross-section'
        Prefix for plot titles
    plot_amplitude : bool, default=True
        Whether to plot amplitude
    plot_phase : bool, default=True
        Whether to plot phase
    figsize : tuple, optional
        Figure size, auto-calculated if None
    """
    # determine number of subplots
    n_plots = plot_amplitude + plot_phase
    if n_plots == 0:
        raise ValueError("At least one of plot_amplitude or plot_phase must be True")
    
    # auto-calculate figsize
    if figsize is None:
        figsize = (6 * n_plots + 4, 6)
    
    # create subplots
    if n_plots > 1:
        fig, axs = plt.subplots(1, n_plots, figsize=figsize)
        axs = axs.flatten()
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axs = [ax]
    
    # create title parts
    title_parts = [title_prefix]
    if alpha_test is not None and delta_test is not None:
        title_parts.append(f'Cross-section: alpha = {alpha_test}, delta = {delta_test}')
    
    if used_ranges:
        title_parts.append(f'Selected Ranges: {used_ranges}')
    
    ax_index = 0
    
    # amplitude
    if plot_amplitude:
        ax = axs[ax_index]
        ax_index += 1
        
        # plot
        ax.scatter(filtered_data[baseline_col], filtered_data[amplitude_col],
                   alpha=0.65, s=8, c='lightblue',
                   label=f'Original ({len(filtered_data):,} points)')
        
        ax.scatter(filtered_data[baseline_col], results_amplitude,
                   alpha=0.7, s=25, c='yellow', edgecolors='orange',
                   linewidth=1.0, marker='s',
                   label=f'Predictions ({len(filtered_data):,} points)')
        
        # add range markers
        if used_ranges:
            for start, stop in used_ranges:
                ax.axvline(start, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
                ax.axvline(stop, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # configure
        ax.set_xlabel(f'{baseline_col}, (Earth Diameters)', fontsize=12)
        ax.set_ylabel('Visibility Amplitude', fontsize=12)
        ax.set_title('\n'.join(title_parts), fontsize=13, pad=15)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='upper right')
    
    # phase
    if plot_phase:
        ax = axs[ax_index] if n_plots > 1 else axs[0]
        
        if results_phase is None:
            raise ValueError("results_phase must be provided when plot_phase=True")
        
        # plot
        ax.scatter(filtered_data[baseline_col], filtered_data[phase_col],
                   alpha=0.65, s=8, c='lightblue',
                   label=f'Original ({len(filtered_data):,} points)')
        
        ax.scatter(filtered_data[baseline_col], results_phase,
                   alpha=0.7, s=25, c='yellow', edgecolors='orange',
                   linewidth=1.0, marker='s',
                   label=f'Predictions ({len(filtered_data):,} points)')
        
        # add range markers
        if used_ranges:
            for start, stop in used_ranges:
                ax.axvline(start, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
                ax.axvline(stop, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # configure
        ax.set_xlabel(f'{baseline_col}, (Earth Diameters)', fontsize=12)
        ax.set_ylabel('Visibility Phase (radians)', fontsize=12)
        ax.set_title('\n'.join(title_parts), fontsize=13, pad=15)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    return fig, axs

def plot_residuals_analysis(filtered_data, results_amplitude, results_phase,
                            baseline_col='Baseline',
                            amplitude_col='Amplitude',
                            phase_col='Phase', alpha_test=None, delta_test=None,
                            figsize=(16, 8), return_stats=False):
    """
    Comprehensive residual analysis with multiple plots and statistics.
    
    Parameters:
    -----------
    results_amplitude : array-like
        Amplitude predictions
    results_phase : array-like, optional
        Direct phase predictions. If None, calculated from results_denormalized
    results_denormalized : array-like, optional
        Denormalized predictions [Re, Im] for phase calculation
    phase_col : str, optional
        Column name for phase in filtered_data. If None, calculated from Re/Im
    return_stats : bool, default=False
        If True, return dictionary with all statistics
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with residual analysis plots
    stats : dict (optional)
        Dictionary with all calculated statistics if return_stats=True
    """
    # prepare data
    results_amplitude = np.asarray(results_amplitude).flatten()
    
    # calculate amplitude residuals and statistics
    amplitude_residuals = filtered_data[amplitude_col] - results_amplitude
    rmse_amplitude = np.sqrt(np.mean(amplitude_residuals ** 2))
    mae_amplitude = np.mean(np.abs(amplitude_residuals))
    
    # prepare phase data
    results_phase = np.asarray(results_phase).flatten()
    
    # calculate phase residuals
    phase_residuals = filtered_data[phase_col] - results_phase
    rmse_phase = np.sqrt(np.mean(phase_residuals ** 2))
    mae_phase = np.mean(np.abs(phase_residuals))
    
    # create figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs = axs.flatten()
    
    # title parts
    title_parts = []
    if alpha_test is not None and delta_test is not None:
        title_parts.append(f'alpha = {alpha_test}, delta = {delta_test}')
    
    # amplitude residuals vs baseline
    axs[0].scatter(filtered_data[baseline_col], amplitude_residuals,
                   alpha=0.6, s=10, c='blue')
    axs[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axs[0].set_xlabel(f'{baseline_col}, (Earth Diameters)')
    axs[0].set_ylabel('Amplitude Residual')
    axs[0].set_title(f'Amplitude Residuals (RMSE: {rmse_amplitude:.5f})')
    axs[0].grid(True, alpha=0.3)
    
    # phase residuals vs baseline
    axs[1].scatter(filtered_data[baseline_col], phase_residuals,
                   alpha=0.6, s=10, c='green')
    axs[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axs[1].set_xlabel(f'{baseline_col}, (Earth Diameters)')
    axs[1].set_ylabel('Phase Residual (radians)')
    axs[1].set_title(f'Phase Residuals (RMSE: {rmse_phase:.5f})')
    axs[1].grid(True, alpha=0.3)
    
    # amplitude residuals histogram
    axs[2].hist(amplitude_residuals, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axs[2].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axs[2].set_xlabel('Amplitude Residual')
    axs[2].set_ylabel('Frequency')
    axs[2].set_title(f'Amplitude Residual Distribution\nMean: {amplitude_residuals.mean():.5f}, '
                     f'Std: {amplitude_residuals.std():.5f}')
    axs[2].grid(True, alpha=0.3)
    
    # phase residuals histogram
    axs[3].hist(phase_residuals, bins=50, alpha=0.7, color='green', edgecolor='black')
    axs[3].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axs[3].set_xlabel('Phase Residual (radians)')
    axs[3].set_ylabel('Frequency')
    axs[3].set_title(f'Phase Residual Distribution\nMean: {phase_residuals.mean():.5f}, '
                     f'Std: {phase_residuals.std():.5f}')
    axs[3].grid(True, alpha=0.3)
    
    # main title
    if title_parts:
        fig.suptitle(f'Residual Analysis - {title_parts[0]}', fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"Residuals statistics:\n")
    print(f"Amplitude:")
    print(f"- RMSE: {rmse_amplitude:.5f}")
    print(f"- MAE: {mae_amplitude:.5f}")
    print(f"- Mean residual: {amplitude_residuals.mean():.5f}")
    print(f"- Std residual: {amplitude_residuals.std():.5f}\n")
    print(f"Phase:")
    print(f"- RMSE: {rmse_phase:.5f}")
    print(f"- MAE: {mae_phase:.5f}")
    print(f"- Mean residual: {phase_residuals.mean():.5f}")
    print(f"- Std residual: {phase_residuals.std():.5f}")
    
    if return_stats:
        stats = {
            'rmse_amplitude': rmse_amplitude,
            'mae_amplitude': mae_amplitude,
            'amplitude_residuals_mean': amplitude_residuals.mean(),
            'amplitude_residuals_std': amplitude_residuals.std(),
            'amplitude_residuals': amplitude_residuals,
            'rmse_phase': rmse_phase,
            'mae_phase': mae_phase,
            'phase_residuals_mean': phase_residuals.mean(),
            'phase_residuals_std': phase_residuals.std(),
            'phase_residuals': phase_residuals
        }
        return fig, stats
    
    return fig