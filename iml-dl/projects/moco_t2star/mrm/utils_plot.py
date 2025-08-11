import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Rectangle


def individual_imshow(data, title=None, save_path=None, vmin=None, vmax=None,
                      replace_nan=False, show_colorbar=False, cmap='gray',
                      text_left=None, text_right=None, fontsize=28, pad=0,
                      box_params=None):

    if len(data.shape) == 1:
        data = np.repeat(data[np.newaxis], 40, axis=0).T
    elif len(data.shape) > 2:
        raise ValueError("Data must be 1D or 2D.")

    fig, ax = plt.subplots(figsize=(2, 2*data.shape[0]/data.shape[1]))
    add_imshow_axis(ax=ax, data=data, vmin=vmin, vmax=vmax, title=title,
                    show_colorbar=show_colorbar, cmap=cmap,
                    text_left=text_left, text_right=text_right,
                    replace_nan=replace_nan, fontsize=fontsize)
    if box_params:
        x, y = box_params.get('xy', (0, 0))
        width = box_params.get('width', 10)
        height = box_params.get('height', 10)
        edgecolor = box_params.get('edgecolor', 'white')
        linewidth = box_params.get('linewidth', 1)
        rect = Rectangle((x, y), width, height, linewidth=linewidth,
                         edgecolor=edgecolor, facecolor='none')
        ax.add_patch(rect)
    plt.tight_layout(pad=pad)
    plt.show()

    if save_path:
        fig.savefig(save_path.replace(".png", ".svg"))


def add_imshow_axis(ax, data, vmin=None, vmax=None, title=None,
                    show_colorbar=False, cmap='gray',
                    text_left=None, text_right=None, text_mid=None,
                    replace_nan=False, fontsize=28, box_params=None):
    if replace_nan:
        data[data == np.nan] = 200
    if vmin is not None and vmax is not None:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(data, cmap=cmap)
    if title:
        ax.text(0.5, 0.9, title, fontsize=fontsize, color='white', ha='center',
                va='center', transform=ax.transAxes)
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=fontsize)
    if text_left:
        ax.text(0.02, 0, text_left, color='white', fontsize=fontsize, ha='left',
                va='bottom', transform=ax.transAxes)
    if text_mid:
        ax.text(0.5, 0, text_mid, color='white', fontsize=fontsize, ha='center',
                va='bottom', transform=ax.transAxes)
    if text_right:
        ax.text(0.98, 0, text_right, color='white', fontsize=fontsize, ha='right',
                va='bottom', transform=ax.transAxes)
    if box_params:
        x, y = box_params.get('xy', (0, 0))
        width = box_params.get('width', 10)
        height = box_params.get('height', 10)
        edgecolor = box_params.get('edgecolor', 'white')
        linewidth = box_params.get('linewidth', 1)
        rect = Rectangle((x, y), width, height, linewidth=linewidth,
                         edgecolor=edgecolor, facecolor='none')
        ax.add_patch(rect)
    ax.axis('off')



def plot_violin_iq_metrics(metrics_merged, experiments, statistical_tests=None,
                           p_value_threshold=0.05, save_individual_plots=True,
                           save_path=None, ylims=None):
    metrics_dict = {
        "mae": "MAE [ms]",
        "ssim": "SSIM",
        "fsim": "FSIM",
        "lpips": "LPIPS"
    }
    imgs_dict = {
        "img_motion": "Motion-\ncorrupted",
        "AllSlices": "All \nslices",
        "Proposed": "Proposed\nPHIMO",
        "NoKeepCenter": "No \nKeepCenter",
        "img_hrqr": "HR/QR",
        "AllSlices-NoKeepCenter": "Unoptimized\nPHIMO",
        "img_orba": "ORBA",
        "img_sld": "SLD",
    }
    color_dict = {
        "img_motion": ["#BEBEBE", 0.7],
        "img_orba": ["#8497B0", 0.8],
        "img_sld": ["#005293", 0.75],
        "img_hrqr": ["#44546A", 0.9],
        "AllSlices-NoKeepCenter": ["#A9C09A", 0.8],
        "Proposed": ["#6C8B57", 0.9]
    }

    figsize = (12, 6) if save_individual_plots else (16, 6)
    fig, axs = plt.subplots(2, len(metrics_merged), figsize=figsize, sharey='col')
    positions = np.arange(len(experiments))
    colors = [color_dict[e][0] for e in experiments]
    alphas = [color_dict[e][1] for e in experiments]

    motion_types = list(metrics_merged[list(metrics_merged.keys())[0]].keys())

    for idx, m_key in enumerate(metrics_merged.keys()):
        ylim = None
        if ylims is not None:
            if ylims[m_key] is not None:
                ylim = ylims[m_key]
        data_stronger = [
            np.reshape(metrics_merged[m_key][motion_types[0]][exp_name], (-1,)) for
            exp_name in experiments]
        if len(motion_types) > 1:
            data_minor = [
                np.reshape(metrics_merged[m_key][motion_types[1]][exp_name], (-1,)) for
                exp_name in experiments]
        else:
            data_minor = None

        if data_stronger and data_stronger[0].size > 0:
            vp = configure_axis_iqm(axs[0, idx], data_stronger, positions, colors,
                                    alphas, metrics_dict[m_key],
                                    [imgs_dict[e] for e in experiments],
                                    ylim=ylim)
            vp_range = list(get_violin_heights(vp))
            if isinstance(ylim, list):
                if vp_range[0] > ylim[1] - 0.1 * (ylim[1] - ylim[0]):
                    vp_range[0] = ylim[1] - 0.1 * (ylim[1] - ylim[0])
                if vp_range[1] < ylim[0] + 0.1 * (ylim[1] - ylim[0]):
                    vp_range[1] = ylim[0] + 0.1 * (ylim[1] - ylim[0])

            # Add significance brackets for "stronger" data
            if statistical_tests is not None:
                p_values_stronger = np.array(
                    statistical_tests[m_key][motion_types[0]]["p"])
                comb_stronger = np.array(
                    statistical_tests[m_key][motion_types[0]]["comb"])
                show_brackets(p_values_stronger, comb_stronger, positions,
                              violin_ranges=vp_range, color='dimgrey', top=True,
                              p_value_threshold=p_value_threshold, ax=axs[0, idx])

        if data_minor and data_minor[0].size > 0:
            vp = configure_axis_iqm(axs[1, idx], data_minor, positions, colors,
                                    alphas, metrics_dict[m_key],
                                    [imgs_dict[e] for e in experiments],
                                    ylim=ylim)
            vp_range = get_violin_heights(vp)

            # Add significance brackets for "minor" data
            if statistical_tests is not None:
                p_values_minor = np.array(statistical_tests[m_key][motion_types[1]]["p"])
                comb_minor = np.array(statistical_tests[m_key][motion_types[1]]["comb"])
                show_brackets(p_values_minor, comb_minor, positions,
                              violin_ranges=vp_range, color='dimgrey', top=True,
                              p_value_threshold=p_value_threshold, ax=axs[1, idx])

    fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.07,
                        wspace=0.35, hspace=0.3)
    plt.show()
    if save_individual_plots:
        if save_path is None:
            ValueError("A save path must be provided to save individual plots.")
        save_individual_subplots(fig, axs, save_dir=save_path, prefix="violin_")


def configure_axis_iqm(ax, data, positions, colors, alphas, ylabel, experiments, ylim=None):
    """Configures the violin plot axis with provided properties."""
    vp = ax.violinplot(data, positions=positions, widths=0.4, showmeans=True,
                       showmedians=False, showextrema=False)

    # Set colors for violin bodies
    for v, col, al in zip(vp['bodies'], colors, alphas):
        v.set_facecolor(col)
        v.set_edgecolor(col)
        v.set_alpha(al)
        v.set_zorder(2)

    vp['cmeans'].set_edgecolor("#E8E8E8")
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlim(-0.3, len(experiments) - 0.7)
    if ylim is not None:
        # check if it's a list:
        if isinstance(ylim, list) and len(ylim) == 2:
            ax.set_ylim(top=ylim[1], bottom=ylim[0])
        else:
            ax.set_ylim(top=ylim)
    ax.set_xticks(positions)
    ax.set_xticklabels([], fontsize=10)
    ax.tick_params(axis='y', labelsize=14)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.grid(True, which='major', color='lightgray', linestyle='--', zorder=0)
    ax.yaxis.grid(True, which='minor', color='lightgray', linestyle='--', zorder=0)

    return vp


def save_individual_subplots(fig, axs, save_dir=".", prefix="subplot"):
    """
    Saves each subplot in the provided figure as an individual image.

    Parameters:
    - fig: matplotlib.figure.Figure, the complete figure containing subplots.
    - axs: list of lists, containing the axes for each subplot.
    - save_dir: str, the directory where individual subplots will be saved.
    - prefix: str, prefix for saved files to identify them.
    """
    # Loop through each subplot
    for row in range(axs.shape[0]):
        for col in range(axs.shape[1]):
            # Hide all other axes except the one being saved
            for r in range(axs.shape[0]):
                for c in range(axs.shape[1]):
                    if (r, c) != (row, col):
                        axs[r, c].set_visible(False)

            # Save the specific subplot
            fig.savefig(f"{save_dir}/{prefix}_row{row}_col{col}.svg", dpi=300,
                        bbox_inches='tight')

            # Restore visibility for all subplots for the next iteration
            for r in range(axs.shape[0]):
                for c in range(axs.shape[1]):
                    axs[r, c].set_visible(True)


def get_violin_heights(violin_object):
    """Get the heights and bottom of the violin plots."""

    top_values, bottom_values = [], []
    for violin in violin_object['bodies']:
        path = violin.get_paths()[0]
        y_data = path.vertices[:, 1]
        top_values.append(np.max(y_data))
        bottom_values.append(np.min(y_data))

    return max(top_values), min(bottom_values)


def show_brackets(p_cor, ind, violins, violin_ranges, color='dimgrey',
                  top=True, p_value_threshold=0.05, ax=None, ylim=None):
    """Show non-overlapping brackets dynamically positioned above/below violins."""

    if ax is None:
        raise ValueError("An axis (ax) must be provided for plotting brackets.")

    # Determine plot height and base position for brackets
    y_min, y_max = ylim if ylim else ax.get_ylim()
    plot_height = y_max - y_min
    offset = 0.02 * plot_height if top else -0.02 * plot_height
    base_height = violin_ranges[0 if top else 1] + offset

    # Get and sort significant pairs
    significant_pairs = sorted(
        (pair for pair, p in zip(ind, p_cor) if p >= p_value_threshold),
        key=lambda x: min(x))

    # Plot non-overlapping brackets
    for i, (v1, v2) in enumerate(significant_pairs):
        bracket_height = calculate_bracket_height(base_height, i, plot_height,
                                                  top)
        barplot_annotate_brackets(violins[v1], violins[v2], bracket_height,
                                  0.02 * plot_height, color, 13, top=top,
                                  label="N", ax=ax)


def calculate_bracket_height(base_height, index, plot_height, top):
    """Calculate height for each bracket to avoid overlap."""

    return base_height + (
        index * 0.1 * plot_height if top else -index * 0.1 * plot_height)


def barplot_annotate_brackets(num1, num2, y, bracket_height, color, font_size,
                              top, label="", ax=None):
    """Annotate bar plot with brackets and text."""

    if ax is None:
        raise ValueError("An axis (ax) must be provided for plotting brackets.")

    # Coordinates for bracket and text
    bracket_y = [y, y + bracket_height, y + bracket_height, y] if top else [
        y + bracket_height, y, y, y + bracket_height]
    text_y = y + bracket_height if top else y - bracket_height
    va = 'bottom' if top else 'top'

    # Draw bracket and label
    ax.plot([num1, num1, num2, num2], bracket_y, c=color)
    ax.text((num1 + num2) / 2, text_y, label, ha='center', va=va,
            fontsize=font_size, color=color)


def plot_violin_line_det_metrics(line_detection_metrics, experiment_ids, subjects,
                                 statistical_tests=None, p_value_threshold=0.05,
                                 metric_type="correctly_excluded", save_path=None):
    # Extract metric values for each experiment and subject
    data = {exp_id: [] for exp_id in experiment_ids}
    for exp_id in experiment_ids:
        for subject in subjects:
            data[exp_id].extend(line_detection_metrics[metric_type][exp_id][subject])

    metric_dict = {
        "correctly_excluded": "Correctly Excluded Lines",
        "wrongly_excluded": "Wrongly Excluded Lines",
        "mae_masks": "MAE",
        "accuracy": "Accuracy"
    }
    color_dict = {
        "img_motion": ["#BEBEBE", 0.7],
        "img_orba": ["#8497B0", 0.8],
        "img_sld": ["#005293", 0.75],
        "img_hrqr": ["#44546A", 0.9],
        "AllSlices-NoKeepCenter": ["#A9C09A", 0.8],
        "Proposed": ["#6C8B57", 0.9]
    }

    positions = np.arange(len(experiment_ids))
    cols = [color_dict[e][0] for e in experiment_ids]
    alphas = [color_dict[e][1] for e in experiment_ids]

    fig, ax = plt.subplots(figsize=(4, 3))
    vp = ax.violinplot([data[exp_id] for exp_id in experiment_ids], positions=positions, widths=0.4, showmeans=True, showmedians=False, showextrema=False)
    for v, col, al in zip(vp['bodies'], cols, alphas):
        v.set_facecolor(col)
        v.set_edgecolor(col)
        v.set_alpha(al)
        v.set_zorder(2)
    vp['cmeans'].set_edgecolor("#E8E8E8")
    vp_range = get_violin_heights(vp)

    if statistical_tests is not None:
        p_values_stronger = np.array(
            statistical_tests[metric_type]["p"])
        comb_stronger = np.array(
            statistical_tests[metric_type]["comb"])
        show_brackets(p_values_stronger, comb_stronger, positions,
                      violin_ranges=vp_range, color='dimgrey', top=True,
                      p_value_threshold=p_value_threshold, ax=ax)

    ax.set_ylabel(metric_dict[metric_type], fontsize=18)
    ax.set_xlim(-0.5, len(experiment_ids) - 0.5)
    ax.set_xticks(positions, experiment_ids, fontsize=20)
    ax.set_xticklabels([], fontsize=10)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='y', labelsize=16)

    ax.yaxis.grid(True, which='major', color='lightgray', linestyle='--')
    ax.yaxis.grid(True, which='minor', color='lightgray', linestyle='--')
    plt.subplots_adjust(left=0.23, right=0.93)
    if save_path is not None:
        # fig.savefig(save_path, dpi=300, bbox_inches='tight')
        fig.savefig(save_path.replace(".png", ".svg"), dpi=300, bbox_inches='tight')
    plt.show()

    # Split subjects into three groups based on substrings "low", "mid", and "high"
    groups = {'low': [], 'mid': [], 'high': []}
    for subject in subjects:
        if 'low' in subject:
            groups['low'].append(subject)
        elif 'mid' in subject:
            groups['mid'].append(subject)
        elif 'high' in subject:
            groups['high'].append(subject)

    # Create separate violin plots for each group
    for group_name, group_subjects in groups.items():
        if group_subjects:
            group_data = {exp_id: [] for exp_id in experiment_ids}
            for exp_id in experiment_ids:
                for subject in group_subjects:
                    group_data[exp_id].extend(line_detection_metrics["correctly_excluded"][exp_id][subject])

            fig, ax = plt.subplots(figsize=(7, 6))
            vp = ax.violinplot([group_data[exp_id] for exp_id in experiment_ids], positions=positions, widths=0.4, showmeans=True, showmedians=False, showextrema=False)
            for v, col in zip(vp['bodies'], cols):
                v.set_facecolor(col)
                v.set_edgecolor(col)
                v.set_alpha(0.7)
                v.set_zorder(2)
            vp['cmeans'].set_edgecolor("#E8E8E8")
            ax.set_ylabel("Correctly Excluded Lines", fontsize=20)
            ax.set_title(f'{group_name.capitalize()} Subjects', fontsize=20)
            # ax.set_ylim(0, 1)
            ax.set_xlim(-0.5, len(experiment_ids) - 0.5)
            ax.set_xticks(positions, experiment_ids, fontsize=20)
            ax.set_yticks(np.arange(0, 1.1, 0.2), fontsize=20)
            ax.minorticks_on()
            ax.set_yticks(np.arange(0, 1.1, 0.1), minor=True)
            ax.tick_params(axis='y', labelsize=20)
            ax.yaxis.grid(True, which='major', color='lightgray', linestyle='--')
            ax.yaxis.grid(True, which='minor', color='lightgray', linestyle='--')
            plt.show()


def plot_pr_curves(precision_recall, experiment_ids, save_path=None,
                   positive_prevalence= None):

    color_dict = {
        "img_motion": ["#BEBEBE", 0.7],
        "img_orba": ["#8497B0", 0.8],
        "img_sld": ["#005293", 0.75],
        "img_hrqr": ["#44546A", 0.9],
        "AllSlices-NoKeepCenter": ["#A9C09A", 0.8],
        "Proposed": ["#6C8B57", 0.9]
    }

    cols = [color_dict[e][0] for e in experiment_ids]
    alphas = [color_dict[e][1] for e in experiment_ids]

    fig, ax = plt.subplots(figsize=(4, 3))
    for exp_id, col, al in zip(experiment_ids, cols, alphas):
        ax.plot(precision_recall["recall"][exp_id], precision_recall["precision"][exp_id],
                label=exp_id, color=col, alpha=al)
    if positive_prevalence is not None:
        ax.axhline(y=positive_prevalence, color="gray", linestyle="--",
                   linewidth=1.5,
                   label=f"Random Classifier (P={positive_prevalence:.2f})")
    ax.set_xlabel("Recall", fontsize=22)
    ax.set_ylabel("Precision", fontsize=22)
    ax.set_yticks(np.arange(0.2, 1.1, 0.2))
    ax.minorticks_on()
    ax.tick_params(axis='both', labelsize=20)
    ax.yaxis.grid(True, which='major', color='lightgray', linestyle='--')
    ax.yaxis.grid(True, which='minor', color='lightgray', linestyle='--')
    ax.xaxis.grid(True, which='major', color='lightgray', linestyle='--')
    ax.xaxis.grid(True, which='minor', color='lightgray', linestyle='--')
    plt.subplots_adjust(left=0.23, right=0.93)
    if save_path is not None:
        fig.savefig(save_path.replace(".png", ".svg"), dpi=300, bbox_inches='tight')
    plt.show()


def plot_precision_recall_kspace_loc(line_detection_metrics, experiment_ids,
                                     subjects, save_path=None,
                                     statistical_tests=None, p_value_threshold=0.05):

    color_dict = {
        "img_motion": ["#BEBEBE", 0.7],
        "img_orba": ["#8497B0", 0.8],
        "img_sld": ["#005293", 0.75],
        "img_hrqr": ["#44546A", 0.9],
        "AllSlices-NoKeepCenter": ["#A9C09A", 0.8],
        "Proposed": ["#6C8B57", 0.9]
    }
    metric_dict = {
        "precision": "Precision",
        "recall": "Recall"
    }
    cols = [color_dict[e][0] for e in experiment_ids]
    alphas = [color_dict[e][1] for e in experiment_ids]
    positions = np.linspace(0.5, 1.5, len(experiment_ids))

    for metric_type in ["precision", "recall"]:
        data_central = {exp_id: [] for exp_id in experiment_ids}
        for exp_id in experiment_ids:
            for subject in subjects:
                data_central[exp_id].extend(
                    line_detection_metrics[metric_type+"_central"][exp_id][subject])
        data_peripheral = {exp_id: [] for exp_id in experiment_ids}
        for exp_id in experiment_ids:
            for subject in subjects:
                data_peripheral[exp_id].extend(
                    line_detection_metrics[metric_type+"_peripheral"][exp_id][subject])

        fig, ax = plt.subplots(figsize=(4, 3))
        vp = add_violins(ax, [data_central[exp_id] for exp_id in experiment_ids],
                         positions, cols, alphas)
        if statistical_tests is not None:
            p_values_stronger = np.array(
                statistical_tests[metric_type+"_central"]["p"])
            comb_stronger = np.array(
                statistical_tests[metric_type+"_central"]["comb"])
            show_brackets(p_values_stronger, comb_stronger, positions,
                          violin_ranges=get_violin_heights(vp), color='dimgrey',
                          top=True, ax=ax,
                          p_value_threshold=p_value_threshold)

        vp = add_violins(ax, [data_peripheral[exp_id] for exp_id in experiment_ids],
                            positions+2, cols, alphas)
        if statistical_tests is not None:
            p_values_stronger = np.array(
                statistical_tests[metric_type+"_peripheral"]["p"])
            comb_stronger = np.array(
                statistical_tests[metric_type+"_peripheral"]["comb"])
            show_brackets(p_values_stronger, comb_stronger, positions+2,
                          violin_ranges=get_violin_heights(vp), color='dimgrey',
                          top=True, ax=ax,
                          p_value_threshold=p_value_threshold)

        ax.set_ylabel(metric_dict[metric_type], fontsize=22)
        ax.minorticks_on()
        ax.tick_params(axis='y', labelsize=20)
        ax.yaxis.grid(True, which='major', color='lightgray', linestyle='--')
        ax.yaxis.grid(True, which='minor', color='lightgray', linestyle='--')
        ax.set_xticks([1, 3])
        ax.set_xticklabels(["Central\n25%", "Peripheral\n75%"], fontsize=20)
        ax.tick_params(axis='x', which='minor', length=0)
        plt.subplots_adjust(left=0.23, right=0.93)
        if save_path is not None:
            fig.savefig(save_path.replace(".png", ".svg").replace("XXX", metric_type),
                        dpi=300, bbox_inches='tight')
        plt.show()


def scatter_cloud(y_values, x_value, ax, col):

    ax.scatter(x_value+np.random.uniform(-0.1, 0.1, size=len(y_values)), y_values,
               s=8, alpha=0.6, color=col)


def add_violins(ax, data, positions, colors, alphas):
    vp = ax.violinplot(data, positions=positions, widths=0.2, showmeans=True,
                       showmedians=False, showextrema=False)
    for v, col, al in zip(vp['bodies'], colors, alphas):
        v.set_facecolor(col)
        v.set_edgecolor(col)
        v.set_alpha(al)
        v.set_zorder(2)
    vp['cmeans'].set_edgecolor("#E8E8E8")
    return vp


def plot_class_differences(deviations, experiment_ids, save_path=None, mode="abs"):

    color_dict = {
        "img_motion": ["#BEBEBE", 0.7],
        "img_orba": ["#8497B0", 0.8],
        "img_sld": ["#005293", 0.75],
        "img_hrqr": ["#44546A", 0.9],
        "AllSlices-NoKeepCenter": ["#A9C09A", 0.8],
        "Proposed": ["#6C8B57", 0.9]
    }

    transform = abs if mode == "abs" else lambda x: x

    cols = [color_dict[e][0] for e in experiment_ids]
    alphas = [color_dict[e][1] for e in experiment_ids]
    positions = np.linspace(0.5, 1.5, len(experiment_ids))

    fig, ax = plt.subplots(figsize=(4, 3))
    vp = add_violins(ax, [transform(deviations["motion_free"][exp_id]) for exp_id in experiment_ids],
                        positions, cols, alphas)
    vp = add_violins(ax, [transform(deviations["motion_corrupted"][exp_id]) for exp_id in experiment_ids],
                        positions+2, cols, alphas)

    ax.set_xticks([1, 3], ["1", "0"], fontsize=20)
    ax.set_xlabel("Reference Class", fontsize=20)
    if mode == "abs":
        ax.set_ylabel("Absolute Error\n(Linewise)", fontsize=20)
    else:
        ax.set_ylabel("Prediction - Reference\n(Linewise)", fontsize=20)
    ax.minorticks_on()
    ax.tick_params(axis='both', labelsize=20)
    ax.yaxis.grid(True, which='major', color='lightgray', linestyle='--')
    ax.yaxis.grid(True, which='minor', color='lightgray', linestyle='--')
    plt.subplots_adjust(left=0.23, right=0.93)
    if save_path is not None:
        fig.savefig(save_path.replace(".png", ".svg"), dpi=300, bbox_inches='tight')
    plt.show()


def adapt_font_size(fig_width):

    small_fontsize = 34
    large_fontsize = 38
    base_width = 10

    return small_fontsize / base_width * fig_width, large_fontsize / base_width * fig_width


def plot_line_det_metrics(line_detection_metrics, slices_ind,
                          experiment_ids, subjects,
                          metric_type, split_by_level=True,
                          save_path=None):
    colors = ["#6C8B57", "tab:gray"]
    exp_dicts = {
        "Proposed": "Even/Odd (Proposed)",
        "AllSlices": "Individual Slices"
    }

    if split_by_level:
        # Split subjects into three groups based on substrings "low", "mid", and "high"
        groups = {'low': [], 'mid': [], 'high': []}
        for subject in subjects:
            if 'low' in subject:
                groups['low'].append(subject)
            elif 'mid' in subject:
                groups['mid'].append(subject)
            elif 'high' in subject:
                groups['high'].append(subject)

        for group_name, group_subjects in groups.items():
            if group_subjects:
                # Extract metric values for each experiment and subject
                num_slices = len(next(iter(
                    line_detection_metrics[metric_type][experiment_ids[0]].values())))
                avg_metrics = {exp_id: np.zeros(num_slices) for exp_id in experiment_ids}
                std_metrics = {exp_id: np.zeros(num_slices) for exp_id in experiment_ids}

                for exp_id in experiment_ids:
                    for slice_idx in range(num_slices):
                        slice_values = []
                        for subject in group_subjects:
                            value = line_detection_metrics[metric_type][exp_id][subject][slice_idx]
                            slice_values.append(value)
                        avg_metrics[exp_id][slice_idx] = np.mean(slice_values)
                        std_metrics[exp_id][slice_idx] = np.std(slice_values)

                # Create line plot
                fig, ax = plt.subplots(figsize=(10, 6))
                for exp_id, color in zip(experiment_ids, colors):
                    ax.errorbar(range(num_slices), avg_metrics[exp_id],
                                yerr=std_metrics[exp_id], label=exp_id, color=color,
                                capsize=5)

                ax.set_xlabel("Slice Number", fontsize=14)
                ax.set_ylabel(f"Average {metric_type.replace('_', ' ').capitalize()}", fontsize=14)
                ax.set_title(
                    f"Average {metric_type.replace('_', ' ').capitalize()} per Slice - {group_name.capitalize()} Subjects",
                    fontsize=16)
                ax.legend(fontsize=12)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.show()

    # Plot for all subjects
    all_slices = np.unique(np.concatenate([slices_ind[sub] for sub in slices_ind.keys()]))
    avg_metrics = {exp_id: np.zeros(len(all_slices)) for exp_id in experiment_ids}
    std_metrics = {exp_id: np.zeros(len(all_slices)) for exp_id in experiment_ids}

    for exp_id in experiment_ids:
        for nr, slice_idx in enumerate(all_slices):
            slice_values = []
            for subject in subjects:
                try:
                    ind = np.where(slices_ind[subject] == slice_idx)[0][0]
                    value = line_detection_metrics[metric_type][exp_id][subject][ind]
                    slice_values.append(value)
                except IndexError:
                    continue
            if slice_values:
                avg_metrics[exp_id][nr] = np.mean(slice_values)
                std_metrics[exp_id][nr] = np.std(slice_values)

    # Create line plot for all subjects
    fig, ax = plt.subplots(figsize=(10, 10))
    sfs, lfs = adapt_font_size(10)
    for exp_id, color in zip(experiment_ids, colors):
        ax.plot(all_slices, avg_metrics[exp_id], label=exp_dicts[exp_id],
                color=color, linewidth=6)
        ax.fill_between(all_slices, avg_metrics[exp_id] - std_metrics[exp_id],
                        avg_metrics[exp_id] + std_metrics[exp_id], color=color,
                        alpha=0.1)

    ax.set_xlabel("Slice Number", fontsize=lfs)
    ax.set_ylabel(f"Line Detection {metric_type.replace('_', ' ').capitalize()}",
                  fontsize=lfs)
    legend = ax.legend(fontsize=sfs, loc="lower right")
    for handle in legend.legendHandles:
        if handle.get_label() == "Even/Odd (Proposed)":
            handle.set_alpha(0.8)
        handle.set_linewidth(12.0)

    ax.tick_params(axis='both', which='major', labelsize=sfs)
    ax.grid(True, which='both', linestyle='--', linewidth=0.8)
    plt.subplots_adjust(top=0.98, bottom=0.12,
                        left=0.18, right=0.96)
    if save_path is not None:
        fig.savefig(save_path.replace(".png", ".svg"), dpi=300,
                    bbox_inches='tight')
    plt.show()


def plot_violin_line_det_freq_dep(line_detection_metrics, exps, save_path=None):

    descrs = ["accuracy_lowfreq", "accuracy_medfreq", "accuracy_highfreq"]
    positions = np.linspace(1, 3, 3)/2
    if len(exps) == 2:
        positions = np.linspace(1, 3, 3)/3*2 - 0.1
    elif len(exps) > 2:
        print("Only one or two experiments are supported.")
    fig, ax = plt.subplots(figsize=(5, 5))
    sfs, lfs = adapt_font_size(5)
    vp = add_violins(ax, [list(line_detection_metrics[descr][exps[0]].values()) for descr in descrs],
                     positions=positions,
                     colors=(["#BEBEBE", "#BEBEBE", "#BEBEBE"] if len(exps) == 2
                             else ["#6C8B57", "#6C8B57", "#6C8B57"]),
                     alphas=[0.7, 0.7, 0.7, 0.7])
    for v in vp['bodies']:
        v.set_zorder(2)
    if len(exps) == 2:
        vp = add_violins(ax, [list(line_detection_metrics[descr][exps[1]].values()) for descr in descrs],
                         positions=positions+0.2,
                         colors=["#6C8B57", "#6C8B57", "#6C8B57"],
                         alphas=[0.9, 0.9, 0.9, 0.9])
        for v in vp['bodies']:
            v.set_zorder(2)
        positions = positions + 0.1
    plt.xticks(positions, ["low", "medium", "high"], fontsize=sfs)
    plt.xlabel("k-space frequencies", fontsize=lfs)
    ax.tick_params(axis='y', which='major', labelsize=sfs)
    plt.ylabel("Line Detection Accuracy", fontsize=lfs)
    plt.ylim(0.6, 1.02)
    ax.yaxis.grid(True, which='both', linestyle='--', linewidth=0.8, zorder=0)
    plt.subplots_adjust(left=0.25, right=0.98, top=0.95, bottom=0.2)
    if save_path is not None:
        fig.savefig(save_path.replace(".png", ".svg"), dpi=300,
                    bbox_inches='tight')
    plt.show()


def grid_imshow_center_recons(img_zf, img_zf_motion, echoes, save_path=None, cmap='gray'):

    n_cols = len(echoes)
    n_rows = 2
    img_height, img_width = abs(img_zf[0, echoes[0]]).T.shape
    aspect_ratio = img_height / img_width
    fig_width = n_cols * 5
    fig_height = fig_width * (n_rows * aspect_ratio) / n_cols + 5
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(fig_width, fig_height),
                             gridspec_kw={'wspace': 0, 'hspace': 0.08})
    sfs, lfs = adapt_font_size(fig_width)

    for col, echo in enumerate(echoes):
        data_top = abs(img_zf[0, echo]).T
        data_bottom = abs(img_zf_motion[0, echo]).T
        vmin = np.min([np.nanmin(data_top), np.nanmin(data_bottom)])
        vmax = np.max([np.nanmax(data_top), np.nanmax(data_bottom)])
        ax_top = axes[0, col] if n_cols > 1 else axes[0]
        im_top = ax_top.imshow(data_top, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_top.axis('off')
        ax_bottom = axes[1, col] if n_cols > 1 else axes[1]
        im_bottom = ax_bottom.imshow(data_bottom, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_bottom.axis('off')

        fig.text(0.5 / n_cols + col / n_cols, 0.5, f"Echo {echo}", fontsize=sfs,
                 ha='center', va='center')

    fig.text(0.5, 0.94, "Without intentional motion", fontsize=lfs, ha='center', va='center')
    fig.text(0.5, 0.06, "With intentional motion", fontsize=lfs, ha='center', va='center')

    fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.09, wspace=0, hspace=0.1)
    if save_path:
        fig.savefig(save_path, dpi=400)
    plt.show()


def plot_violin_loss_functions(loss, descrs, save_path=None, loss_2=None):

    positions = np.linspace(1, len(descrs), len(descrs))/2
    if loss_2 is not None:
        positions = np.linspace(1, len(descrs), len(descrs))/3*2 - 0.1
    fig, ax = plt.subplots(figsize=(5, 5))
    sfs, lfs = adapt_font_size(5)
    vp = add_violins(ax, [loss[descr] for descr in descrs],
                     positions=positions,
                     colors=["#BEBEBE", "#BEBEBE", "#BEBEBE", "#BEBEBE"],
                     alphas=[0.7, 0.7, 0.7, 0.7])
    if loss_2 is not None:
        vp = add_violins(ax, [loss_2[descr] for descr in descrs],
                         positions=positions+0.2,
                         colors=["#6C8B57", "#6C8B57", "#6C8B57", "#6C8B57"],
                         alphas=[0.9, 0.9, 0.9, 0.9])
        for v in vp['bodies']:
            v.set_zorder(2)
        positions = positions + 0.1
    plt.xticks(positions, ["motion-\nfree", "low", "medium", "high"], fontsize=sfs)
    plt.xlabel("Motion level", fontsize=lfs)
    ax.tick_params(axis='y', which='major', labelsize=sfs)
    plt.ylabel("$L_{phys}$", fontsize=lfs)
    ax.yaxis.grid(True, which='both', linestyle='--', linewidth=0.8, zorder=0)
    plt.subplots_adjust(left=0.25, right=0.98, top=0.95, bottom=0.2)
    if save_path is not None:
        fig.savefig(save_path.replace(".png", ".svg"), dpi=300,
                    bbox_inches='tight')
    plt.show()
