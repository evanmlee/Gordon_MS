import pandas as pd 
import numpy as np
import os 
import re 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings
import scipy.stats as sp_stats

import matplotlib
matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('font', serif='Helvetica Neue') 
matplotlib.rc('text', usetex='false') 

######=====================================================================######
####                    Experiment Color Palette Declarations                ####
######=====================================================================######
MG01_bar_palette={"A":"#FEACA7","B":"#D4D4D4"}
MG01_point_palette={"A":"#FF2804","B":"#000000"}
MG01_legend_labels = {"A":"Preotella","B":"No Prevotella"}

MG02_bar_palette ={"A":"#FEACA7","C":"#D4D4D4","D":"#9FC9EB"}
MG02_point_palette={"A":"#FF2804","C":"#000000","D":"#3E58A8"}
MG02_legend_labels={"A":"$\it{P. copri}$, MDCF-2","C":"No $\it{P. copri}$, MDCF-2","D":"$\it{P. copri}$, Mirpur-18"}
######=====================================================================######
####                    Significance testing helper functions                ####
######=====================================================================######
def sig_str_from_p(pval,sigstr_fmt="std",ns_str=""):
    """Takes a p-value and generates a corresponding significance string. 
    
    If sigstr_fmt = "std": the returned string is of the format "*","**", etc. 
    If sigstr_fmt = "pval": the returned string is of the format "P={pval}"
    ns_str is the string used when p >= 0.05 
    """
    if sigstr_fmt == "std":
        if pval < 0.001:
            return "***"
        elif pval < 0.01:
            return "**"
        elif pval < 0.05:
            return "*"
        else:
            return ns_str
    elif sigstr_fmt == "pval":
        if pval < 0.05 and pval > 0.001:
            return "P={0}".format(pval)
        elif pval < 0.001: 
            return "P<0.001"
        else: 
            return "n.s." 
    else: 
        raise ValueError("Unrecognized sigstr_fmt: {0}. Please use 'std' or 'pval'.".format(sigstr_fmt))

def pairwise_subgroup_stats_testing(group_data_df,analyte_col,subgroup_col,stats_test="Mann-Whitney",sigstr_fmt="std",
                            ns_str=""):
    """Does pairwise statistical significance testing for each unnique pair of subgroups from group_data_df for a given analyte
    and returns the results as a DataFrame. 
    
    @param group_data_df (pd.DataFrame): DataFrame containing at least analyte_col (numeric mass spec data for that analyte), 
    as well as subgroup_col, which should contain unique values for each subgroup to be tested. group_data can be a subset 
    of samples from an entire dataset (i.e. only the samples for a given Group value/ treatment, for which subgroup testing
    can then be done for sex or cage number).
    @param analyte_col (str): Corresponds to a column label in group_data_df which should contain numeric MS data which will
    be tested for statistically significant differences between subgroups
    @param subgroup_col (str): Corresponds to a column label in group_data_df which should contain unique metadata values for each
    subgroup to be tested (i.e. cage numbers such as AF1, AF2, etc. which will be found in multiple records). 
    @param sigstr_fmt (str): Passed to sig_str_from_p, dictates whether a asterisk-form or P = X.XX significance string will be given 
    in the results DataFrame
    @param ns_str (str): Passed to sig_str_from_p, dictates default value for tests with a P >= 0.05 

    @return comparisons_stats_df (pd.DataFrame): columns ["Analyte","stat","pval","sig_str","Comparison"]. Each row will be 
    for one subgroup pairwise comparison and statistical test, with the analyte tested, the test statistic and test p-value, 
    a correspondign significance string for the p-value, and a string denoting the subgroups compared for that test as the 
    values in that row. The number of rows will be generated generically by (n_subgroups)*(n_subgroups-1)//2. 
    """

    stats_columns = ["Analyte","stat","pval","sig_str","Comparison"]
    #Get unique values of subgroups in subgroup_col and use to calculate n_comparisons
    subgroups = group_data_df[subgroup_col].unique().tolist()
    n_subgroups = len(subgroups)
    n_comparisons = (n_subgroups)*(n_subgroups-1)//2
    #Initialize empty DataFrame for results of statistical tests 
    comparisons_stats_df = pd.DataFrame(index=range(n_comparisons),columns=stats_columns)
    #Case handling by the number of unique subgroups; currently n_subgroups <= 4 are supported 
    if n_subgroups == 1:
        raise ValueError("There is only one unique subgroup in subgroup_col and pairwise statistical testing cannot be done.")
    elif n_subgroups == 2:
        sg_1, sg_2 = subgroups[0],subgroups[1]
        #One pairwise comparison: 1-2
        comparison_str = "{0}_{1}".format(sg_1,sg_2)
        sg_1_data = group_data_df[group_data_df[subgroup_col]==sg_1]
        sg_2_data = group_data_df[group_data_df[subgroup_col]==sg_2]
        if stats_test == "Mann-Whitney":
            stat, pval = sp_stats.mannwhitneyu(x=sg_1_data[analyte_col],
                                            y=sg_2_data[analyte_col],alternative="two-sided")
        sig_str = sig_str_from_p(pval,sigstr_fmt=sigstr_fmt,ns_str=ns_str)
        comparisons_stats_df.loc[0,:] = dict(zip(stats_columns,[analyte_col,stat,pval,sig_str,comparison_str]))
    elif n_subgroups == 3:
        sg_1, sg_2, sg_3 = subgroups[0],subgroups[1],subgroups[2]
        #Three pairwise comparisons: 1-2, 2-3, 1-3
        for i,first_subgroup, second_subgroup in zip(range(n_comparisons),
                                                    [sg_1,sg_2,sg_1],
                                                    [sg_2,sg_3,sg_3]):
            comparison_str = "{0}_{1}".format(first_subgroup,second_subgroup)
            first_subgroup_data = group_data_df[group_data_df[subgroup_col]==first_subgroup]
            second_subgroup_data = group_data_df[group_data_df[subgroup_col]==second_subgroup]
            if stats_test == "Mann-Whitney":
                stat, pval = sp_stats.mannwhitneyu(x=first_subgroup_data[analyte_col],y=second_subgroup_data[analyte_col],
                            alternative="two-sided")
            sig_str = sig_str_from_p(pval,sigstr_fmt=sigstr_fmt,ns_str=ns_str)
            comparisons_stats_df.loc[i,:] = dict(zip(stats_columns,[analyte_col,stat,pval,sig_str,comparison_str]))
    elif n_subgroups >= 4:
        if n_subgroups > 4: 
            warnings.warn("Have not implemented 5+ pairwise compairsons yet; results returned are for the first 4 subgroups.")
        sg_1, sg_2, sg_3, sg_4 = subgroups[0],subgroups[1],subgroups[2],subgroups[3]
        #Six pairwise comparisons: 1-2, 2-3, 3-4, 1-3, 2-4, 1-4
        for i,first_subgroup, second_subgroup in zip(range(n_comparisons),
                                                    [sg_1,sg_2,sg_3,sg_1,sg_2,sg_1],
                                                    [sg_2,sg_3,sg_4,sg_3,sg_4,sg_4]):
            comparison_str = "{0}_{1}".format(first_subgroup,second_subgroup)
            first_subgroup_data =  group_data_df[group_data_df[subgroup_col]==first_subgroup]
            second_subgroup_data = group_data_df[group_data_df[subgroup_col]==second_subgroup]
            if stats_test == "Mann-Whitney":
                stat, pval = sp_stats.mannwhitneyu(x=first_subgroup_data[analyte_col],
                                y=second_subgroup_data[analyte_col],alternative="two-sided")
            sig_str = sig_str_from_p(pval,sigstr_fmt=sigstr_fmt,ns_str=ns_str)
            comparisons_stats_df.loc[i,:] = dict(zip(stats_columns,[analyte_col,stat,pval,sig_str,comparison_str]))
    return comparisons_stats_df

def single_analyte_stats(data_df,analyte_col,tissue,sampleID_col,group_re=r'([ABCD])',stats_test="Mann-Whitney",
                        split="group",sigstr_fmt="std",ns_str=""):
    """For a specified analyte in data_df, do statistical significance testing between up to 4 groups. 

    Wrapper function for calls to pairwise_subgroup_stats_testing based on the value of split. 
    Returns a DataFrame with columns: "Analyte", "stat", "pval", "sig_str", "Comparison"
    Analyte - MS analyte for which significance testing was done; stat, pval, sig_str are results of corresponding stats_test;
    Comparison - details which groups (extracted by group_re) which were compared for that statistical test 
    
    """
    #Universal processing - if tissue is specified and is in the columns, subset the data to only that tissue.  
    if "Tissue" in data_df.columns:
        tissue_data_df = data_df.loc[data_df["Tissue"]==tissue] #subset to tissue if tissue in data_df columns, otherwise ignore  
        if len(tissue_data_df) >= 0:
            data_df = tissue_data_df
    else:
        pass 
    #If "Group" column is not already provided, use group_re to extract it from the sampleID_col. 
    if "Group" not in data_df.columns: 
            data_df["Group"] = data_df[sampleID_col].str.extract(group_re)
    groups = data_df["Group"].unique()
    n_groups = len(groups)
    stats_columns = ["Analyte","stat","pval","sig_str","Comparison"]
    if split == "group":
        #For group pariwise testing, use entire data_df and split subgroups for pairwise testing by "Group" column
        single_analyte_stats_df = pairwise_subgroup_stats_testing(data_df,analyte_col,subgroup_col="Group",
                                    stats_test=stats_test,sigstr_fmt=sigstr_fmt,ns_str=ns_str)
    elif split == "cluster":
        assert "Cluster" in data_df.columns,"split='cluster' but 'Cluster' column not provided in data_df"
        #For Cluster pariwise testing, do test batchwise for each unique experiment Group (on the assumption that clusters are 
        #within each Group); concatenate results for each group into final returned single_analyte_stats_df 
        single_analyte_stats_df = pd.DataFrame(columns=stats_columns)
        for group in groups: 
            group_data = data_df[data_df["Group"]==group]
            group_clusters_comparisons_stats = pairwise_subgroup_stats_testing(group_data,analyte_col,subgroup_col="Cluster",
                                                stats_test=stats_test,sigstr_fmt=sigstr_fmt,ns_str=ns_str)
            single_analyte_stats_df = pd.concat((single_analyte_stats_df,group_clusters_comparisons_stats))
    elif split == "sex":
        #For sex pairwise testing, test differences between male and female within each experimental group; concatenate results 
        #for each group into final returned single_analyte_stats_df 
        single_analyte_stats_df = pd.DataFrame(columns=stats_columns)
        for group in groups: 
            group_data = data_df[data_df["Group"]==group]
            group_clusters_comparisons_stats = pairwise_subgroup_stats_testing(group_data,analyte_col,subgroup_col="Sex",
                                                stats_test=stats_test,sigstr_fmt=sigstr_fmt,ns_str=ns_str)
            single_analyte_stats_df = pd.concat((single_analyte_stats_df,group_clusters_comparisons_stats))
    elif split == "cage_number":
        #For cage_number pairwise testing, test differences between each pair of cages within each experimental group; 
        #concatenate results for each group into final returned single_analyte_stats_df 
        # single_analyte_stats_df = pd.DataFrame(index=stats_columns)
        single_analyte_stats_df = pd.DataFrame(columns=stats_columns)
        for group in groups: 
            group_data = data_df[data_df["Group"]==group]
            group_clusters_comparisons_stats = pairwise_subgroup_stats_testing(group_data,analyte_col,subgroup_col="Cage_Number",
                                                stats_test=stats_test,sigstr_fmt=sigstr_fmt,ns_str=ns_str)
            single_analyte_stats_df = pd.concat((single_analyte_stats_df,group_clusters_comparisons_stats))
    return single_analyte_stats_df

def multiple_analyte_stats(data_df,analytes,tissues,sampleID_col,group_re=r'([ABCD])',stats_test="Mann-Whitney",
                            sigstr_fmt="std",ns_str="",split="group"):
    stats_columns = ["Analyte","Tissue","stat","pval","sig_str","Comparison"]
    multiple_analyte_stats_df = pd.DataFrame(columns=stats_columns)
    for analyte in analytes:
        for tissue in tissues: 
            single_analyte_stats_df = single_analyte_stats(data_df,analyte,tissue,sampleID_col,
                                                    group_re=group_re,stats_test=stats_test,split=split)
            multiple_analyte_stats_df = pd.concat((multiple_analyte_stats_df,single_analyte_stats_df))
    return stats_df

######=====================================================================######
####                            DataFrame Reformatting                      ####
######=====================================================================######

def tissue_analyte_long_df(data_df,analytes,tissues,sampleID_col,group_re=r'([ABCD])'):
    """Generate long form DataFrame with Measurement (numerical) and Analyte, Tissue, Group (categorical) columns.

    """
    long_columns = ["Measurement","Analyte","Tissue","Group"]
    if "Group" not in data_df.columns: #extract group information if not present in data_df
        data_df["Group"] = data_df[sampleID_col].str.extract(group_re)
    groups = data_df["Group"].unique()
    #Handling for kmeans or other cluster labeling and partitioning of data 
    optional_columns = ["Sex","Cage_Number","Cluster"]
    for optional_col in optional_columns:
        if optional_col in data_df.columns:
            long_columns.extend([optional_col])
    long_df = pd.DataFrame(columns=long_columns)
    for tissue in tissues: 
        for analyte in analytes: 
            #Partition input dataframe
            if "Tissue" in data_df.columns: 
                tissue_data = data_df.loc[data_df["Tissue"]==tissue] #important for multiple analytes/tissues case
            else:
                tissue_data = data_df #single analyte case (usually)
            
            tissue_short_df = pd.DataFrame(index=tissue_data.index,columns=long_df.columns)
            #Populate long form data for seaborn 
            for group in groups: 
                group_data = tissue_data[tissue_data["Group"]==group]
                tissue_short_df.loc[group_data.index,"Measurement"] = group_data[analyte]
                tissue_short_df.loc[group_data.index,"Group"] = group_data["Group"]

                for optional_col in optional_columns: 
                    if optional_col in long_columns: #If previously checked to be in data, populate data in short/long_df
                        tissue_short_df.loc[group_data.index,optional_col] = group_data[optional_col]

                tissue_short_df.loc[group_data.index,"Analyte"] = pd.Series(index=group_data.index,data=[analyte] * len(group_data))
                tissue_short_df.loc[group_data.index,"Tissue"] = pd.Series(index=group_data.index,data=[tissue] * len(group_data))
                if "Cluster" in data_df.columns:  
                    tissue_short_df.loc[group_data.index,["Cluster"]] = group_data.loc[:,"Cluster"]
            long_df = pd.concat([long_df,tissue_short_df],ignore_index=True)
    long_df.dropna(how="all",inplace=True)        
    return long_df


######=====================================================================######
####                        Analyte Bar/Swarm Plots                          ####  
######=====================================================================######
#Single metabolite barplot generation 
def single_analyte_barplot(data_df,analyte,tissue,sampleID_col="SampleID",
                           ax=None,units="µM",legend=True,stats_test="Mann-Whitney",sigstr_fmt="std",
                           fig_fmt="pdf",split="group",figures_dir="figures",group_re="([ABCD])",
                           bar_palette=MG01_bar_palette,point_palette=MG01_point_palette,legend_labels=MG01_legend_labels,title_str=""):
    # data_df["Group"] = data_df[sampleID_col].str.extract(group_re)
    #Partition input dataframe
    #stats test results 
    #single_analyte_stats_df = single_analyte_stats(data_df,analyte,tissue,sampleID_col=sampleID_col,
                                              #stats_test=stats_test,split=split,group_re=group_re)
    #Generate long form data for seaborn 
    tissue_analyte_df = tissue_analyte_long_df(data_df,[analyte],[tissue],sampleID_col=sampleID_col,
                                               group_re=group_re)
    if not ax:
        new_fig = True
        if split == "cage_number":
            fig,ax = plt.subplots(1,1,figsize=(3,6))
        else:
            fig,ax = plt.subplots(1,1,figsize=(1,6))
    else: 
        new_fig = False

    #Overlay swarmplot over barplot of individual sample data 
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if split == "group":
            sns.barplot(data=tissue_analyte_df,x="Group",y="Measurement",ax=ax,zorder=0,palette=bar_palette,
                        capsize=0.1,errwidth=1)
            sns.swarmplot(data=tissue_analyte_df,x="Group",y="Measurement", ax=ax, zorder=1,palette=point_palette)
        elif split == "cluster":
            sns.barplot(data=tissue_analyte_df,x="Cluster",y="Measurement",ax=ax,zorder=0,palette=bar_palette,
                        capsize=0.1,errwidth=1)
            sns.swarmplot(data=tissue_analyte_df,x="Cluster",y="Measurement", ax=ax, zorder=1,palette=point_palette)
        elif split == "sex":
            sns.barplot(data=tissue_analyte_df,x="Group",y="Measurement",hue="Sex",palette=bar_palette,ax=ax,zorder=0,
                        capsize=0.1,errwidth=1)
            sns.swarmplot(data=tissue_analyte_df,x="Group",y="Measurement",hue="Sex", palette=point_palette,ax=ax, zorder=1,dodge=True)
        elif split == "cage_number":
            sns.barplot(data=tissue_analyte_df,x="Cage_Number",y="Measurement",hue="Group",palette=bar_palette,ax=ax,zorder=0,
                        capsize=0.1,errwidth=1)
            sns.swarmplot(data=tissue_analyte_df,x="Cage_Number",y="Measurement",hue="Group", palette=point_palette,ax=ax, zorder=1,dodge=True)
    #spaghetti code for changing error bar hues on seaborns barplot 
    face_colors = list(bar_palette.values())
    point_colors = list(point_palette.values())
#     lines_per_err = 3
#     for i, line in enumerate(ax.get_lines()):
#         newcolor = point_colors[i//lines_per_err]
#         line.set_color(newcolor)
    #Matplotlib formatting 
    #Add units to ylabel
    ax.set_ylabel("{0} ({1})".format(analyte,units),fontsize=12,fontweight="bold")
    ax.set_xlabel(tissue.replace("_"," "),labelpad=10,fontsize=12,fontweight="bold")
    #plot_significance_brackets_single_analyte(ax,single_analyte_stats_df)
    #bold xtick/yticks
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontweight('bold') for label in labels]
    ax.set_xticks([])#Remove xticks (will just show A/B group labels)
    
    if new_fig:
        #Add title, legend to new plot 
        if not title_str:
            title_str = analyte 
        ax.set_title(title_str.replace("_"," "),fontsize=14,fontweight="bold")
        if legend:
            patches = []
            if type(legend_labels) == dict:
                legend_labels = legend_labels.values()
            for facecolor, legend_label in zip(face_colors,legend_labels):
                group_patch = matplotlib.patches.Patch(color=facecolor, label=legend_label)
                patches.append(group_patch)
            plt.legend(handles=patches)
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1)) 
        
        analyte_path_str = analyte.replace('/','.').replace(":",".")
        fig_fname = "{0}_{1}.{2}".format(analyte_path_str,tissue,fig_fmt)
        figure_path = os.path.join(figures_dir,fig_fname)
        os.makedirs(figures_dir,exist_ok=True)
        plt.savefig(figure_path,dpi=300,bbox_inches="tight",facecolor="w")
        
        
def multiple_analyte_barplot(data_df,analytes,tissues,group_re="([ABCD])",sampleID_col="SampleID",
                           ax=None,units="µM",stats_test="Mann-Whitney",legend=False,sigstr_fmt="std",
                           fig_fmt="pdf"):
    """Not fully updated yet :(
    """
    if not ax:
        plot_width = (len(analytes)*len(tissues))
        fig,ax = plt.subplots(1,1,figsize=(plot_width,6))
    long_df = tissue_analyte_long_df(data_df,analytes,tissues,sampleID_col=sampleID_col,group_re=group_re)
    stats_df = multiple_analyte_stats(data_df,analytes,tissues,sampleID_col=sampleID_col,
                                       group_re=group_re,stats_test=stats_test)
    pvals, sig_strs = stats_df["pval"].tolist(),stats_df["sig_str"].tolist()
    #Overlay swarmplot over barplot of individual sample data 
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if len(tissues) > 1: 
            sns.barplot(data=long_df,x="Tissue",y="Measurement",hue="Group",ax=ax,zorder=0,palette=HWDC_bar_palette,
                    capsize=0.1,errwidth=1)
            sns.swarmplot(data=long_df,x="Tissue",y="Measurement",hue="Group",ax=ax, zorder=1,
                          palette=HWDC_point_palette,dodge=True)
            ax.set_xlabel("Tissue",labelpad=5,fontsize=16,weight="bold")
            ax.set_ylabel("{0} ({1})".format(analytes[0],units),fontsize=16,weight="bold")
        else: 
            sns.barplot(data=long_df,x="Analyte",y="Measurement",hue="Group",ax=ax,zorder=0,palette=HWDC_bar_palette,
                    capsize=0.1,errwidth=1)
            sns.swarmplot(data=long_df,x="Analyte",y="Measurement",hue="Group",ax=ax, zorder=1,
                          palette=HWDC_point_palette,dodge=True)
            if units == "nM":
                ax.set_ylabel("Acylcarnitine levels ({0})".format(units),fontsize=16,weight="bold")
            elif units == "µM":
                ax.set_ylabel("Amine levels ({0})".format(units),fontsize=16,weight="bold")
            ax.set_xlabel(tissues[0],labelpad=0,fontsize=16,weight="bold")

    #spaghetti code for changing error bar hues on seaborns barplot 
    face_colors = list(HWDC_bar_palette.values())
    point_colors = list(HWDC_point_palette.values())
    lines_per_err = 3
    for i, line in enumerate(ax.get_lines()):
        color_index = i//(lines_per_err*len(tissues)*len(analytes))
        newcolor = point_colors[color_index]
        line.set_color(newcolor)
    #Matplotlib formatting 
    xtick_labels = ax.get_xticklabels()
    if len(analytes) > 1:
        ax.set_xticklabels(xtick_labels,fontsize=12,weight="bold")
    #bold xtick/yticks
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontweight('bold') for label in labels]
    #Sig String annotation and brackets 
    if len(analytes) > 1:
        items = analytes
    else:
        items = tissues
    xticks = ax.get_xticks()
    ymin,ymax = ax.get_ylim()
    ydiff = ymax-ymin
    y1, y2 = ymax-ydiff*0.01,ymax
    for i,item in enumerate(items):
        x1, x2 = xticks[i]-0.25,xticks[i]+0.25
        pval,sig_str = pvals[i], sig_strs[i]
        significance_bracket(ax,pval,sig_str,sigstr_fmt=sigstr_fmt,x1=x1,x2=x2,y1=y1,y2=y2)
    if legend:
        A_patch = matplotlib.patches.Patch(color=face_colors[0], label='Prevotella')
        B_patch = matplotlib.patches.Patch(color=face_colors[1], label='No Prevotella')
        plt.legend(handles=[A_patch,B_patch])
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1)) 
    else:
        ax.get_legend().remove() #remove legend
    if len(analytes) > 6:
        analytes = analytes[:3] +["..."]+ analytes[-3:]
    tissues_path_str = ".".join(tissues)
    analyte_path_str = ",".join(analytes).replace("/","").replace(":","_")
    fig_fname = "{0}_{1}.{2}".format(analyte_path_str,tissue,fig_fmt)
    figure_path = os.path.join(figures_dir,fig_fname)
    os.makedirs(figures_dir,exist_ok=True)
    plt.savefig(figure_path,dpi=300,bbox_inches="tight",facecolor="w")
    plt.savefig(figure_path,dpi=300,bbox_inches="tight",facecolor="w")

def plot_significance_brackets_single_analyte(ax,single_analyte_stats_df,sigstr_fmt="std"):
    """Plots significance brackets and annotations on ax using the results in single_analyte_stats_df.
    """
    if len(single_analyte_stats_df) == 1:
        pval, sig_str = single_analyte_stats_df["pval"].iloc[0],single_analyte_stats_df["sig_str"].iloc[0]
        significance_bracket(ax,pval,sig_str,sigstr_fmt=sigstr_fmt)
    else: 
        #Nested three pairwise comaprisons significance brackets 
        if len(single_analyte_stats_df) == 3:
            #Use xticks and ylim for bracket positioning
            xticks = ax.get_xticks()
            ymin,ymax = ax.get_ylim()
            ydiff = ymax-ymin
            #y positions for pairwise brackets
            bottom_y1, bottom_y2, top_y1, top_y2  = ymax-(ydiff)*.02, ymax-ydiff*0.01, ymax+(ydiff)*.04,ymax+(ydiff)*.05
            for x1,x2,y1,y2,i in zip([xticks[0],xticks[1],xticks[0]],#x1 of bracket
                                [xticks[1],xticks[2],xticks[2]],#x2 of bracket
                                [bottom_y1,bottom_y1,top_y1],#y1 of bracket
                                [bottom_y2,bottom_y2,top_y2],#y2 of bracket
                                range(len(single_analyte_stats_df))):
                significance_bracket(ax,single_analyte_stats_df["pval"].iloc[i],
                    single_analyte_stats_df["sig_str"].iloc[i],x1=x1,x2=x2,y1=y1,y2=y2)

def significance_bracket(ax,pval,sig_str,sigstr_fmt="std",x1=0,x2=0,y1=0,y2=0):
    #Determining coordinates for brackets and sig_str
    if x1 == 0 and x2 == 0: #Default to first two xticks (ie single analyte situation) if x1/x2 not provided
        xticks = ax.get_xticks()
        x1, x2 = xticks[0], xticks[1]
    if y1 == 0 and y2 == 0:
        ymin, ymax = ax.get_ylim()
        ydiff = ymax-ymin
        y2 = ymax - (ydiff)*0
        y1 = y2-(ydiff)*.01
    #Plot significance bracket and sig_str; different cases for sigstr_fmt (ie ns/* format vs pval=...)
    if sigstr_fmt == "std":
        if sig_str == "":
            sig_str = "ns"
            plt.text((x1+x2)*.5, y2, sig_str, ha='center', va='bottom', color='k',weight="bold")
        elif "*" in sig_str:
            #Positioning for * format sig_str
            plt.text((x1+x2)*.5, y2, sig_str, ha='center', va='bottom', color='k',weight="bold")
        else: 
            plt.text((x1+x2)*.5, y2, sig_str, ha='center', va='bottom', color='k',weight="bold")
        plt.plot([x1,x1, x2, x2], [y1, y2, y2, y1], linewidth=1, color='k')
    elif sigstr_fmt == "pval": #Annotation text is of form p=x.xxx
        pval_str = "p={:.3f}".format(pval)
        plt.text((x1+x2)*.5, y2+ydiff*0.01, pval_str, ha='center', va='bottom', color='k',weight="bold")
        plt.plot([x1,x1, x2, x2], [y1, y2, y2, y1], linewidth=1, color='k')
    else: #Default behavior, don't add bracket/sigstr 
        pass


