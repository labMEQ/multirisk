# --- IMPORTAÇÃO DOS PACOTES ---
import numpy as np
from matplotlib.figure import Figure
import math
import streamlit as st
import pandas as pd
from fpdf import FPDF
import tempfile

st.set_page_config(
    page_title="Multiparametric Risk Calculator",
    layout="wide"
)

# --- APP LAYOUT ---
st.header("MULTIPARAMETRIC RISK CALCULATOR")

# --- Superior panel ---

## Container 0 - "Sample and analysis information"
container_0 = st.write("**Analysis information**")
col1, col2 = st.columns(2)

with col1: 
    st.session_state.sample_name = st.text_input("Sample name")
with col2:
    st.session_state.analysis_date = st.text_input("Date of analysis")

## Container 1 - "Parameters settings": 
container_1 = st.write("**Parameters settings**")
col1, col2 = st.columns(2)

with col1:
    num_par = st.number_input("How many variables should be assessed?", value = 3, min_value = 2)
with col2:
    num_sim = st.number_input("How many iterations should be performed?", value = 100000)

rng_mode = st.segmented_control("Data generated should be random or fixed?", ["Random mode", "Fixed mode"], selection_mode = "single", default = "Random mode")

## Container 2 - "Variables settings":
container_2 = st.tabs(["**Variables settings**"])

variables_values = st.expander("Variables values")

with variables_values:
    col3, col4, col5, col6, col7 = st.columns(5)

    with col3:
        st.write("Variable name")
        var_entries = []
        for i in range(num_par):
            var_entries.append(st.text_input("Variable name", f" Var {i+1}", label_visibility = "collapsed", key = f"nameentry_{i}"))
    with col4:
        st.write("Measured value")
        meas_entries = []
        for i in range(num_par):
            meas_entries.append(st.text_input("Measured value", label_visibility = "collapsed", key = f"measentry_{i}"))
    with col5:
        st.write("Uncertainty")
        unc_entries = []
        for i in range(num_par):
            unc_entries.append(st.text_input("Uncertainty", label_visibility = "collapsed", key = f"uncentry_{i}"))
    with col6:
        st.write("Lower Spec. Lim")
        lsl_entries = []
        for i in range(num_par):
            lsl_entries.append(st.text_input("LSL value", label_visibility = "collapsed", key = f"lslentry_{i}"))
    with col7:
        st.write("Upper Spec. Lim")
        usl_entries = []
        for i in range(num_par):
            usl_entries.append(st.text_input("USL value", label_visibility = "collapsed", key = f"uslentry_{i}"))

correlation_matrix = st.expander("Correlation matrix")

with correlation_matrix:
    st.write("**Correlation Matrix**")

    correl_entries =  np.zeros((num_par, num_par)) # Criação de uma matriz vazia para receber os dados
    
    header_cols = st.columns(num_par + 1)
    header_cols[0].write("")

    for j in range(num_par):
        header_cols[j+1].write(f"**Var {j+1}**")

    for i in range(num_par):
        row_cols = st.columns(num_par + 1)

        row_cols[0].write(f"**Var {i+1}**")

        for j in range(num_par):
            with row_cols[j+1]:
                correl_entries[i, j]  = st.text_input("Correl value", key=f"correlentry_{i}_{j}", value=1.0 if i == j else 0.0,
                    label_visibility="collapsed")

run_calculator = st.button("Save values & compute", use_container_width = True)

## --- Sidebar panel ---

### --- Sidebar.container 1 - About:
side_container1 = st.sidebar.container()

with side_container1:
    st.sidebar.subheader("About the app")
    st.sidebar.write("This app was developed by LabMEQ - Laboratory of Metrology, Examinology and Quality by Design, based at University of São Paulo, Brazil, supervised by Prof. Felipe Rebello Lourenço."
             " The main goal of this app is to provide a simple, reliable and user-friendly algorithm for estimating consumer's risk in pharmaceutical analyses. For further information about the app and examples of its use, please refer to (artigo)")

side_container2 = st.sidebar.container()

with side_container2:
    st.sidebar.subheader("How to use the app")
    st.sidebar.write(
        "Section 0 - Sample and analysis information:"
        " Include a sample name and the date of analysis, if you wish so. This information appears on the PDF report at the end.")
    st.sidebar.write(
        "Section 1 - Parameters settings:"
        " Choose the number of parameters to be assessed and number of simulations to be used. These values must be integers!"
        " Select the mode for generating random numbers: with a random seed (if new dataset on every new simulation run) or with a fixed seed (if same dataset across multiple simulation runs")
    st.sidebar.write(
        "Section 2 - Variable settings:"
        " Once the number of parameters is set, give information about each variable in 'variable values': name (if necessary), measured value, uncertainty, lower and upper specification limits. If a variable has no lower or upper specification limits, you can leave the cell blank.")
    st.sidebar.write(
        "In 'correlation matrix', give information about possible correlation between parameters. These values range between -1.0 and 1.0. Pay attention as this matrix must be symmetric and positive semi-definite.")
    st.sidebar.write(
        "You can then run the simulation, by pressing the button 'Save values & compute'."
        " After the simulations are done, the results are automatically summarized in an interactive table showing the particular and total consumer’s risk values. To support visual interpretation, the program also generates: (a) Histograms showing the simulated measurement distributions for each parameter and the corresponding specification limits; and (b) Scatter plots showing pairwise relationships between parameters, highlighting conforming and non-conforming regions and illustrating the effect of correlations.")
    st.sidebar.write(
        "If you want to, you can also export the results and graphs to a PDF report. By clicking on the 'Generate PDF' button, a PDF archive will be created. To save your report, click the new button 'Download PDF'")
    st.sidebar.write(
        "Important to note: if you do click on the 'Generate PDF' button, the results shown on the interface will disappear, but your data is not missing! The algorithm still stores the results until the PDF archive is created. Only after this is that you will need to rerun the app to get new results.")
    st.sidebar.write("**You can contact us at labmeq@usp.br**")

#---------------------- Multiparametric Risk Assessment----------------------
def Sim_Data(num_par, num_sim, val, unc, correl, seed = None):
    
    mu = np.zeros(num_par)

    if seed is None:
        MVN = np.random.multivariate_normal(mu, correl, num_sim)

    else:
        rng_seed = np.random.default_rng(seed)
        MVN = rng_seed.multivariate_normal(mu, correl, num_sim)
        
    MVN_Data = np.zeros((num_sim, num_par))
    for i in range(num_sim):
        for j in range(num_par):
            MVN_Data[i, j] = val[j] + MVN[i, j] * unc[j]
    return MVN_Data
        
def Multi_Risk(num_par, num_sim, LSL, USL, Data):
    IN = np.zeros(num_par)
    OUT = np.zeros(num_par)
    IN_Total = 0
    OUT_Total = 0

    LSL_arr = np.array(LSL, dtype=float)
    USL_arr = np.array(USL, dtype=float)
    LSL_use = np.where(np.isnan(LSL_arr), -np.inf, LSL_arr)
    USL_use = np.where(np.isnan(USL_arr),  np.inf, USL_arr)

    for i in range(num_sim):
        Total = 0
        for j in range(num_par):
            x = Data[i, j]
            if LSL_use[j] <= x <= USL_use[j]:
                IN[j] += 1
            else:
                OUT[j] += 1
                Total += 1
        if Total == 0:
            IN_Total += 1
        else:
            OUT_Total += 1

    ParticularConsumerRiskValue = OUT / (IN + OUT)
    TotalConsumerRiskValue = OUT_Total / (IN_Total + OUT_Total)

    return {
        "ParticularConsumerRiskValue": ParticularConsumerRiskValue,
        "TotalConsumerRiskValue": TotalConsumerRiskValue,
    }

def _to_float_or_zero(s: str) -> float:
    s = s.strip()
    if s in ("", ".", "+", "-"):
        return 0.0
    return float(s)

def _to_float_or_nan(s: str) -> float:
    s = s.strip()
    if s in ("", ".", "+", "-"):
        return float("nan")
    return float(s)

# --- RESULTS AND GRAPHS ---
# creating file report
def create_pdf(sample_name, analysis_date, variables, correlatix, tableres, fig1, fig2):
    # defining header and footer
    class PDF(FPDF):
        def header(self):
            # Putting logo
            self.image("https://ci3.googleusercontent.com/mail-sig/AIorK4x_wOLCu2Gxtl5wGMbvaD6HQpsaxOLpu_2NFWPqYME0r7bs6Dp6gSdc7fSSNst8roR0W3cKwJq09MFe", 
                       10, 8, 28, type = "png")
            # Putting texts titles
            self.set_font("Helvetica")
            self.set_xy(60, 10)
            self.set_font_size(15)
            self.write_html("""<b>LabMEQ - Laboratório de Metrologia, Examinologia e Qualidade por Design</b>""")
            self.set_xy(60, 22)
            self.set_font_size(12)
            self.write_html("""Faculdade de Ciências Farmacêuticas""")
            self.set_xy(60, 27)
            self.set_font_size(10)
            self.write_html("""Universidade de São Paulo""")
            
        def footer(self):
            # Positioning cursor:
            self.set_y(-15)
            # Setting font:
            self.set_font("Helvetica", style = "I", size = 8)
            # Printing page number:
            self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align = "C")
    
    pdf = PDF()
    pdf.add_page(orientation = "portrait")
    pdf.set_font("Helvetica", size = 12)

    # setting the title, sample name and date of analysis
    pdf.set_y(40)
    pdf.cell(w = 0, h = 10, text = "MULTIPARAMETRIC RISK REPORT", border = 1, new_x = "LMARGIN", new_y = "NEXT", align = "C")
    pdf.cell(95, 10, text = f"Sample name: {st.session_state.sample_name}", border = 1, new_x = "RIGHT", align = "L")
    pdf.cell(95, 10, text = f"Date of analysis: {st.session_state.analysis_date}", border = 1, new_x = "RIGHT", align = "L")
    pdf.ln(20)

    # creating a table from variable values dataframe
    values_df_str = variables.map(str) # converting all data inside dataframe into string type
    V_COLUMNS = [list(values_df_str)] # to get list of dataframe columns
    V_ROWS = values_df_str.values.tolist() # to get list of dataframe rows
    V_DATA = V_COLUMNS + V_ROWS # combining columns and rows in one list
    
    pdf.cell(w = 0, h = 10, text = "Variables values", border = "B", new_x = "LMARGIN", new_y = "NEXT", align = "C")
    pdf.ln(5)
    with pdf.table() as v_table:
        for data_row in V_DATA:
            row = v_table.row()
            for datum in data_row:
                row.cell(datum)
    pdf.ln(10)

    # creating a table from correlation matrix dataframe
    correl_df_str = correlatix.map(str) # converting all data inside dataframe into string type
    C_COLUMNS = [list(correl_df_str)] # to get list of dataframe columns
    C_ROWS = correl_df_str.values.tolist() # to get list of dataframe rows
    C_DATA = C_COLUMNS + C_ROWS # combining columns and rows in one list

    pdf.add_page(orientation = "landscape")
    pdf.set_y(40)
    pdf.cell(w = 0, h = 10, text = "Correlation matrix", border = "B", new_x = "LMARGIN", new_y = "NEXT", align = "C")
    pdf.ln(5)
    
    with pdf.table() as c_table:
        for data_row in C_DATA:
            row = c_table.row()
            for datum in data_row:
                row.cell(datum)
    pdf.ln(10)
        
    # table results
    result_df_str = tableres.map(str) # converting all data inside dataframe into string type
    R_COLUMNS = [list(result_df_str)] # to get list of dataframe columns
    R_ROWS = result_df_str.values.tolist() # to get list of dataframe rows
    R_DATA = R_COLUMNS + R_ROWS # combining columns and rows in one list
    
    pdf.add_page(orientation = "portrait")
    pdf.set_y(40)
    pdf.cell(w = 0, h = 10, text = "Results table", border = "B", new_x = "LMARGIN", new_y = "NEXT", align = "C")
    pdf.ln(5)
    
    with pdf.table() as r_table:
        for data_row in R_DATA:
            row = r_table.row()
            for datum in data_row:
                row.cell(datum)
    pdf.ln(10)
    pdf.write_html(f"""<b> Total Consumer's Risk: {st.session_state.total_risk*100: .3f}%""")
    
    # histogram charts
    pdf.add_page(orientation = "portrait")
    pdf.set_y(40)
    pdf.cell(w = 0, h = 10, txt = "Charts and plots", border = "B", new_x = "LMARGIN", new_y = "NEXT", align = "C")
    pdf.ln(10)
    with tempfile.NamedTemporaryFile(delete = False, suffix = ".png") as tmpfile:
        fig1.savefig(tmpfile.name)
        pdf.image(tmpfile.name, w = pdf.epw)

    # scatter plots
    pdf.add_page(orientation = "portrait")
    pdf.set_y(30)
    with tempfile.NamedTemporaryFile(delete = False, suffix = ".png") as tmpfile:
        fig2.savefig(tmpfile.name)

        if num_par <= 6:            
            pdf.image(tmpfile.name, w = pdf.epw, y = 40)
            
        if num_par > 6:            
            pdf.image(tmpfile.name, h = 230, w = pdf.epw, y = 40)
    
    return bytes(pdf.output())
    
if run_calculator:

    # Session state to retrieve data set by user
    names = [st.session_state[f"nameentry_{i}"] for i in range(num_par)]
    measurev = np.array([_to_float_or_zero(st.session_state[f"measentry_{i}"]) for i in range(num_par)])
    uncertainty  = np.array([_to_float_or_zero(st.session_state[f"uncentry_{i}"]) for i in range(num_par)])
    lsls  = np.array([_to_float_or_nan(st.session_state[f"lslentry_{i}"]) for i in range(num_par)])
    usls  = np.array([_to_float_or_nan(st.session_state[f"uslentry_{i}"]) for i in range(num_par)])

    correl = np.zeros((num_par, num_par), dtype=float)
    for i in range(num_par):
        for j in range(num_par):
            correl[i, j] = _to_float_or_zero(st.session_state[f"correlentry_{i}_{j}"])

    # Data wrangling to new code-variables:
    variable_names = names if names else [f"Var {i+1}" for i in range(num_par)]
    
    measured_values = np.array(measurev, dtype=float)
    
    unc_values = np.array(uncertainty, dtype=float)
    
    LSL_values = np.array(lsls, dtype=float)
    
    USL_values = np.array(usls, dtype=float)

    correl_matrix = np.zeros((num_par, num_par))
    for i in range(num_par):
        for j in range(num_par):
            correl_matrix[i, j] = float(correl[i][j])

    # Definition of new dataframes (to report)

    values_data = {'VARIABLE': variable_names,
                   'MEASURED VALUE': measured_values,
                   'UNCERTAINTY': unc_values,
                   'LOWER LIMIT': LSL_values,
                   'UPPER LIMIT': USL_values}

    st.session_state.values_df = pd.DataFrame(values_data)

    st.session_state.correl_df = pd.DataFrame(correl_matrix, columns = variable_names, index = variable_names)

    # Simulations run
    if rng_mode == "Fixed mode":
        st.session_state.Sim_values = Sim_Data(num_par = num_par, num_sim = num_sim, val = measured_values, unc = unc_values, correl = correl_matrix, seed = 89349648022043)
        result_dict = Multi_Risk(num_par = num_par, num_sim = num_sim, LSL = LSL_values, USL = USL_values, Data = st.session_state.Sim_values)
    
    if rng_mode == "Random mode":
        st.session_state.Sim_values = Sim_Data(num_par = num_par, num_sim = num_sim, val = measured_values, unc = unc_values, correl = correl_matrix)
        result_dict = Multi_Risk(num_par = num_par, num_sim = num_sim, LSL = LSL_values, USL = USL_values, Data = st.session_state.Sim_values)

    if not np.allclose(correl, correl.T, atol = 1e-12):
        st.error("Correlation matrix shows some error. The matrix is not symmetric (i.e. correlation(Var1, Var2) =/= correlation(Var2, Var1)). Please, reevaluate correlation values provided and make sure it is symmetric.")
        st.stop()
                 
    if np.any(np.linalg.eigvalsh(correl) < -1e-12):
        st.error("Correlation matrix shows some error. The matrix is not positive-semidefinite. Please, reevaluate correlation values as it may be mathematically inconsistent.")
        st.stop()
                         
    # Creating new tabs for results and graphs
    results_tab, histogram_tab, scatter_tab = st.tabs(["RESULTS", "HISTOGRAMS", "SCATTER PLOTS"])
    
    with results_tab:
        
        particular_risk = result_dict["ParticularConsumerRiskValue"]
        st.session_state.total_risk = result_dict["TotalConsumerRiskValue"]
        
        st.dataframe(
            {
                "Variable": variable_names,
                "Consumer's Risk (%)": [f"{v*100:.3f}%" for v in particular_risk]})
        total_risk = st.write(f"**Total Consumer's Risk: {st.session_state.total_risk*100: .3f}%**")

        part_risk_df = pd.DataFrame.from_dict(
                    {
                        "Variable": variable_names,
                        "Consumer's Risk (%)": [f"{v*100:.3f}%" for v in particular_risk]})
        
        st.session_state.part_risk_df = part_risk_df
                
    with histogram_tab:

        num_hists = st.session_state.Sim_values.shape[1]
        if variable_names is None:
            variable_names = [f"Var {i+1}" for i in range(num_hists)]
            
        cols = min(3, num_hists) if num_hists > 0 else 1
        rows = math.ceil(num_hists / cols) if num_hists > 0 else 1

        fig_hist = Figure(figsize=(4*cols, 2.6*rows), dpi=100)
        axes_hist = [fig_hist.add_subplot(rows, cols, k+1) for k in range(num_hists)]

        for j, ax in enumerate(axes_hist):
            data = st.session_state.Sim_values[:, j]
            ax.hist(data, bins=30, edgecolor='black')
    
            L = LSL_values[j]; U = USL_values[j]
            ymin, ymax = ax.get_ylim()
            ylab = ymin + 0.92*(ymax - ymin)
    
            if not np.isnan(L):
                ax.axvline(L, linestyle='--', color='black')
                ax.text(L, ylab, "LSL", rotation=90, va='top', ha='right', fontsize=8)
    
            if not np.isnan(U):
                ax.axvline(U, linestyle='--', color='black')
                ax.text(U, ylab, "USL", rotation=90, va='top', ha='left', fontsize=8)
    
            ax.set_title(variable_names[j], fontsize=9)
            ax.tick_params(labelsize=8)
    
        fig_hist.tight_layout()
        st.pyplot(fig_hist)
        st.session_state.fig_hist = fig_hist

    with scatter_tab:
        n = st.session_state.Sim_values.shape[1]
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        num_pairs = len(pairs)
        if num_pairs == 0:
            st.write("Not enough variables for scatter plots.")
            
        cols = min(3, num_pairs)
        rows = math.ceil(num_pairs / cols)
        fig_sc = Figure(figsize=(4*cols, 2.6*rows), dpi=100)
        axes_sc = [fig_sc.add_subplot(rows, cols, k+1) for k in range(num_pairs)]
    
        L_use = np.where(np.isnan(LSL_values), -np.inf, LSL_values)
        U_use = np.where(np.isnan(USL_values),  np.inf, USL_values)
    
        for ax, (i, j) in zip(axes_sc, pairs):
            xi, xj = st.session_state.Sim_values[:, i], st.session_state.Sim_values[:, j]
    
            in_i  = (xi >= L_use[i]) & (xi <= U_use[i])
            in_j  = (xj >= L_use[j]) & (xj <= U_use[j])
            in_ok = in_i & in_j
            out   = ~in_ok
    
            ax.scatter(xi[in_ok], xj[in_ok], s=6, alpha=0.6)
            out_sc = ax.scatter(xi[out],   xj[out],   s=8, alpha=0.8)
            out_sc.set_color('red')
    
            ax.set_xlabel(variable_names[i], fontsize=8)
            ax.set_ylabel(variable_names[j], fontsize=8)
            ax.tick_params(labelsize=8)
    
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            if not np.isnan(LSL_values[i]): xmin = min(xmin, LSL_values[i])
            if not np.isnan(USL_values[i]): xmax = max(xmax, USL_values[i])
            if not np.isnan(LSL_values[j]): ymin = min(ymin, LSL_values[j])
            if not np.isnan(USL_values[j]): ymax = max(ymax, USL_values[j])
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
    
            if not np.isnan(LSL_values[i]):
                ax.axvline(LSL_values[i], linestyle='--', color='black')
                ax.text(LSL_values[i], ymax, "LSL", va='top', ha='right', fontsize=8)
            if not np.isnan(USL_values[i]):
                ax.axvline(USL_values[i], linestyle='--', color='black')
                ax.text(USL_values[i], ymax, "USL", va='top', ha='left', fontsize=8)
            if not np.isnan(LSL_values[j]):
                ax.axhline(LSL_values[j], linestyle='--', color='black')
                ax.text(xmax, LSL_values[j], "LSL", va='bottom', ha='right', fontsize=8)
            if not np.isnan(USL_values[j]):
                ax.axhline(USL_values[j], linestyle='--', color='black')
                ax.text(xmax, USL_values[j], "USL", va='bottom', ha='right', fontsize=8)
    
        fig_sc.tight_layout()
        st.pyplot(fig_sc)
        st.session_state.fig_sc = fig_sc
            
if st.button("Generate PDF"):
    pdf_bytes = create_pdf(sample_name = st.session_state.sample_name,
                           analysis_date = st.session_state.analysis_date,
                           variables = st.session_state.values_df,
                           correlatix = st.session_state.correl_df,
                           tableres = st.session_state.part_risk_df,
                           fig1 = st.session_state.fig_hist,
                           fig2 = st.session_state.fig_sc)
    st.download_button(
        label= "Download PDF",
        data = pdf_bytes,
        file_name = "multirisk_report.pdf",
        mime = "application/pdf")
    
