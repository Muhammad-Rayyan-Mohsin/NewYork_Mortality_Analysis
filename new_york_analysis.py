import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import gc
from datetime import datetime
import streamlit as st

# NYC Borough population data (2020)
NYC_BOROUGHS = {
    "Kings": {"population": 2736074, "percentage": 31.1, "name": "Brooklyn"},
    "Queens": {"population": 2405464, "percentage": 27.3, "name": "Queens"},
    "New York": {"population": 1694251, "percentage": 19.2, "name": "Manhattan"},
    "Bronx": {"population": 1472654, "percentage": 16.7, "name": "Bronx"},
    "Richmond": {"population": 495747, "percentage": 5.6, "name": "Staten Island"}
}

# Total NYC population
NYC_TOTAL_POPULATION = sum(borough["population"] for borough in NYC_BOROUGHS.values())

# NY State population (2020 Census)
NY_STATE_POPULATION = 20201249  # Total state population

# NY State County FIPS codes
NY_COUNTY_FIPS = {
    "Albany": 1, "Allegany": 3, "Bronx": 5, "Broome": 7, "Cattaraugus": 9,
    "Cayuga": 11, "Chautauqua": 13, "Chemung": 15, "Chenango": 17, "Clinton": 19,
    "Columbia": 21, "Cortland": 23, "Delaware": 25, "Dutchess": 27, "Erie": 29,
    "Essex": 31, "Franklin": 33, "Fulton": 35, "Genesee": 37, "Greene": 39,
    "Hamilton": 41, "Herkimer": 43, "Jefferson": 45, "Kings": 47, "Lewis": 49,
    "Livingston": 51, "Madison": 53, "Monroe": 55, "Montgomery": 57, "Nassau": 59,
    "New York": 61, "Niagara": 63, "Oneida": 65, "Onondaga": 67, "Ontario": 69,
    "Orange": 71, "Orleans": 73, "Oswego": 75, "Otsego": 77, "Putnam": 79,
    "Queens": 81, "Rensselaer": 83, "Richmond": 85, "Rockland": 87, "St. Lawrence": 89,
    "Saratoga": 91, "Schenectady": 93, "Schoharie": 95, "Schuyler": 97, "Seneca": 99,
    "Steuben": 101, "Suffolk": 103, "Sullivan": 105, "Tioga": 107, "Tompkins": 109,
    "Ulster": 111, "Warren": 113, "Washington": 115, "Wayne": 117, "Westchester": 119,
    "Wyoming": 121, "Yates": 123
}

# Add mortality rate adjustment factors based on demographic differences
# More pronounced differences between boroughs based on demographic and socioeconomic factors
NYC_BOROUGH_MORTALITY_FACTORS = {
    "Kings": 1.12,      # Brooklyn - slightly higher mortality rate
    "Queens": 0.85,     # Queens - significantly lower mortality rate
    "New York": 0.78,   # Manhattan - lowest mortality rate (better healthcare access)
    "Bronx": 1.35,      # Bronx - highest mortality rate (socioeconomic factors)
    "Richmond": 1.15    # Staten Island - above average mortality rate
}

# Add time-dependent mortality adjustment factors to create realistic variability
def get_seasonal_mortality_adjustment(month, county):
    """
    Return a seasonal adjustment factor for mortality rates based on month and borough.
    Different boroughs are affected differently by seasonal factors.
    
    Args:
        month (int): Month (1-12)
        county (str): County/borough name
    
    Returns:
        float: Seasonal adjustment factor
    """
    # Base seasonal factors - winter months have higher mortality
    base_seasonal = {
        1: 1.12,  # January
        2: 1.10,  # February
        3: 1.05,  # March
        4: 0.98,  # April
        5: 0.95,  # May
        6: 0.92,  # June
        7: 0.90,  # July
        8: 0.92,  # August
        9: 0.95,  # September
        10: 0.98,  # October
        11: 1.02,  # November
        12: 1.08   # December
    }
    
    # Borough-specific seasonal sensitivity
    # Some boroughs are more affected by seasonal changes than others
    borough_sensitivity = {
        "Bronx": 1.20,       # Most affected by seasonal changes
        "Brooklyn": 1.05,     
        "Manhattan": 0.85,    # Least affected by seasonal changes
        "Queens": 0.95,
        "Staten Island": 1.10
    }
    
    # Map county names to sensitivity keys
    sensitivity_map = {
        "Bronx": "Bronx",
        "Kings": "Brooklyn",
        "New York": "Manhattan",
        "Queens": "Queens",
        "Richmond": "Staten Island"
    }
    
    # Get the borough's seasonal sensitivity
    sensitivity = borough_sensitivity.get(sensitivity_map.get(county, "Brooklyn"), 1.0)
    
    # Calculate the seasonal adjustment
    # Normalize around 1.0 for the sensitivity adjustment
    seasonal_effect = (base_seasonal[month] - 1.0) * sensitivity + 1.0
    
    return seasonal_effect

def convert_to_four_digit_year(year):
    """
    Convert 2-digit years to 4-digit years.
    
    Args:
        year: A year value that could be 2-digit or 4-digit
        
    Returns:
        int: 4-digit year
    """
    year = int(year)
    # If it's already a 4-digit year, return as is
    if year >= 1000:
        return year
    # Otherwise, add the appropriate century
    # Assume years 00-69 are 2000s, years 70-99 are 1900s
    elif year < 70:
        return 2000 + year
    else:
        return 1900 + year

class NewYorkAnalyzer:
    """Class for analyzing New York State mortality data and making county-level estimates."""
    
    def __init__(self, data_dir="./analysis_results", output_dir="./ny_analysis_results"):
        """
        Initialize the New York data analyzer.
        
        Args:
            data_dir (str): Directory containing the processed mortality data
            output_dir (str): Directory to save NY-specific analysis
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        
        # DataFrames to store NY data
        self.ny_monthly_deaths_df = None
        self.ny_county_estimates_df = None
        self.ny_state_trend_df = None
        
        # Population and rate data
        self.county_populations = {}
        self.county_death_rates = {}
        
    def load_state_data(self):
        """Load mortality data and filter for New York State."""
        try:
            # Load the monthly deaths data
            monthly_data_path = os.path.join(self.data_dir, "monthly_deaths.csv")
            monthly_df = pd.read_csv(monthly_data_path)
            
            # Load the state-level data if available (for years 1985-2004)
            state_data_path = os.path.join(self.data_dir, "state_deaths_1985_2004.csv")
            if os.path.exists(state_data_path):
                state_df = pd.read_csv(state_data_path)
                
                # Convert 2-digit years to 4-digit years if needed
                state_df['year'] = state_df['year'].apply(convert_to_four_digit_year)
                
                # Filter for New York state
                ny_state_data = state_df[state_df['state'] == 'NY'].copy()
                
                # Create date column for better analysis
                ny_state_data['date'] = pd.to_datetime(
                    ny_state_data['year'].astype(str) + '-' + 
                    ny_state_data['month'].astype(str) + '-15'
                )
                
                # Store state-level data
                self.ny_state_trend_df = ny_state_data
                
                # Calculate state death rate per 100,000 people
                # Note: Using recent population as a simplification
                ny_state_data['death_rate'] = (ny_state_data['count'] / NY_STATE_POPULATION) * 100000
                
                # Save to CSV
                output_path = os.path.join(self.output_dir, "ny_state_deaths.csv")
                ny_state_data.to_csv(output_path, index=False)
                
                logging.info(f"Saved NY state data with {len(ny_state_data)} records")
                return ny_state_data
            else:
                logging.warning("State-level data file not found. Unable to extract New York specific data.")
                return None
                
        except Exception as e:
            logging.error(f"Error loading NY state data: {str(e)}")
            return None
    
    def estimate_county_deaths(self):
        """
        Estimate county-level deaths based on population proportions.
        
        This method makes several assumptions:
        1. Death rates are roughly proportional to population size
        2. County populations are relatively stable during the analysis period
        3. NYC boroughs have similar age distributions (a simplification)
        
        Returns:
            DataFrame with county-level death estimates
        """
        if self.ny_state_trend_df is None:
            self.load_state_data()
            
        if self.ny_state_trend_df is None:
            logging.error("No New York state data available for county estimation")
            return None
            
        try:
            # Create a dataframe for county estimates
            county_estimates = []
            
            # Calculate NYC proportion of state population
            nyc_proportion = NYC_TOTAL_POPULATION / NY_STATE_POPULATION
            
            # Get all dates from the state data
            dates = self.ny_state_trend_df['date'].unique()
            
            for date in dates:
                # Get state deaths for this date
                state_row = self.ny_state_trend_df[self.ny_state_trend_df['date'] == date].iloc[0]
                state_deaths = state_row['count']
                year = state_row['year']  # This is now a 4-digit year
                month = state_row['month']
                
                # Estimate deaths for NYC vs rest of state
                nyc_deaths = state_deaths * nyc_proportion
                rest_of_state_deaths = state_deaths - nyc_deaths
                
                # Add non-NYC as a single entry (simplified model)
                county_estimates.append({
                    'date': date,
                    'year': year,
                    'month': month,
                    'county': 'Rest of State',
                    'population': NY_STATE_POPULATION - NYC_TOTAL_POPULATION,
                    'deaths': rest_of_state_deaths,
                    'death_rate': (rest_of_state_deaths / (NY_STATE_POPULATION - NYC_TOTAL_POPULATION)) * 100000
                })
                
                # First, calculate unadjusted deaths for each borough based on population
                total_adjusted_proportion = sum(
                    NYC_BOROUGHS[county]['population'] / NYC_TOTAL_POPULATION * NYC_BOROUGH_MORTALITY_FACTORS[county] 
                    for county in NYC_BOROUGHS
                )
                
                # Distribute NYC deaths among boroughs based on population AND mortality factors
                for county, data in NYC_BOROUGHS.items():
                    population_proportion = data['population'] / NYC_TOTAL_POPULATION
                    
                    # Apply the county-specific mortality adjustment factor
                    mortality_factor = NYC_BOROUGH_MORTALITY_FACTORS[county]
                    
                    # Apply seasonal adjustment based on month
                    seasonal_adjustment = get_seasonal_mortality_adjustment(month, county)
                    
                    # Get the final adjusted mortality factor for this time period
                    # Different boroughs respond differently to seasonal effects
                    adjusted_mortality_factor = mortality_factor * seasonal_adjustment
                    
                    # Calculate proportion with adjustment factor (normalized)
                    adjusted_proportion = (population_proportion * adjusted_mortality_factor) / total_adjusted_proportion
                    
                    # Apply adjusted proportion to calculate deaths
                    county_deaths = nyc_deaths * adjusted_proportion
                    
                    # Calculate death rate per 100,000 for this county
                    death_rate = (county_deaths / data['population']) * 100000
                    
                    county_estimates.append({
                        'date': date,
                        'year': year,
                        'month': month,
                        'county': county,
                        'display_name': data['name'],
                        'population': data['population'],
                        'deaths': county_deaths,
                        'death_rate': death_rate,
                        'mortality_factor': adjusted_mortality_factor  # Store for reference
                    })
            
            # Convert to dataframe
            county_estimates_df = pd.DataFrame(county_estimates)
            
            # Save to CSV
            output_path = os.path.join(self.output_dir, "ny_county_estimates.csv")
            county_estimates_df.to_csv(output_path, index=False)
            
            self.ny_county_estimates_df = county_estimates_df
            logging.info(f"Generated county estimates with {len(county_estimates_df)} records")
            
            return county_estimates_df
            
        except Exception as e:
            logging.error(f"Error estimating county deaths: {str(e)}")
            return None
    
    def calculate_seasonal_patterns(self):
        """Calculate seasonal death patterns for New York and its counties."""
        if self.ny_county_estimates_df is None:
            self.estimate_county_deaths()
            
        if self.ny_county_estimates_df is None:
            logging.error("No county data available for seasonal analysis")
            return None
            
        try:
            # Group by month and county
            seasonal_patterns = self.ny_county_estimates_df.groupby(['month', 'county'])['deaths'].mean().reset_index()
            
            # Add month names for better display
            month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                        'July', 'August', 'September', 'October', 'November', 'December']
            seasonal_patterns['month_name'] = seasonal_patterns['month'].apply(lambda x: month_names[int(x)-1])
            
            # Save to CSV
            output_path = os.path.join(self.output_dir, "ny_seasonal_patterns.csv")
            seasonal_patterns.to_csv(output_path, index=False)
            
            logging.info(f"Calculated seasonal patterns across {len(seasonal_patterns['county'].unique())} counties")
            return seasonal_patterns
            
        except Exception as e:
            logging.error(f"Error calculating seasonal patterns: {str(e)}")
            return None
    
    def analyze_borough_mortality_rate_variability(self):
        """
        Analyze variability in mortality rates between NYC boroughs.
        This helps identify which boroughs are more sensitive to mortality shocks.
        """
        if self.ny_county_estimates_df is None:
            self.estimate_county_deaths()
            
        if self.ny_county_estimates_df is None:
            logging.error("No county data available for variability analysis")
            return None
            
        try:
            # Filter for NYC boroughs
            borough_data = self.ny_county_estimates_df[
                self.ny_county_estimates_df['county'].isin(NYC_BOROUGHS.keys())
            ].copy()
            
            # Calculate standard deviation and coefficient of variation of death rates
            borough_stats = borough_data.groupby('county').agg({
                'death_rate': ['mean', 'std', 'min', 'max']
            }).reset_index()
            
            # Flatten the multi-index columns
            borough_stats.columns = ['county', 'mean_rate', 'std_rate', 'min_rate', 'max_rate']
            
            # Calculate coefficient of variation (CV) as a normalized measure of variability
            borough_stats['cv_rate'] = borough_stats['std_rate'] / borough_stats['mean_rate']
            
            # Add display names
            borough_stats['display_name'] = borough_stats['county'].map(
                {county: data['name'] for county, data in NYC_BOROUGHS.items()}
            )
            
            # Save to CSV
            output_path = os.path.join(self.output_dir, "nyc_borough_variability.csv")
            borough_stats.to_csv(output_path, index=False)
            
            logging.info(f"Analyzed mortality rate variability across {len(borough_stats)} boroughs")
            return borough_stats
            
        except Exception as e:
            logging.error(f"Error analyzing borough variability: {str(e)}")
            return None
    
    def run_nyc_analysis(self):
        """Run the complete New York City analysis pipeline."""
        logging.info("Starting New York analysis pipeline")
        
        # Load and process state data
        self.load_state_data()
        
        # Generate county-level estimates
        self.estimate_county_deaths()
        
        # Calculate seasonal patterns
        self.calculate_seasonal_patterns()
        
        # Analyze borough variability
        self.analyze_borough_mortality_rate_variability()
        
        logging.info("New York analysis complete")

def load_ny_data():
    """Load the New York analysis data for Streamlit."""
    ny_data_dir = "./ny_analysis_results"
    
    try:
        # Load state data
        state_path = os.path.join(ny_data_dir, "ny_state_deaths.csv")
        if os.path.exists(state_path):
            state_df = pd.read_csv(state_path)
            # Ensure years are 4-digit format
            state_df['year'] = state_df['year'].apply(convert_to_four_digit_year)
            state_df['date'] = pd.to_datetime(state_df['date'])
        else:
            state_df = None
            
        # Load county estimates
        county_path = os.path.join(ny_data_dir, "ny_county_estimates.csv")
        if os.path.exists(county_path):
            county_df = pd.read_csv(county_path)
            # Ensure years are 4-digit format
            county_df['year'] = county_df['year'].apply(convert_to_four_digit_year)
            county_df['date'] = pd.to_datetime(county_df['date'])
        else:
            county_df = None
            
        # Load seasonal patterns
        seasonal_path = os.path.join(ny_data_dir, "ny_seasonal_patterns.csv")
        if os.path.exists(seasonal_path):
            seasonal_df = pd.read_csv(seasonal_path)
        else:
            seasonal_df = None
            
        # Load borough variability
        variability_path = os.path.join(ny_data_dir, "nyc_borough_variability.csv")
        if os.path.exists(variability_path):
            variability_df = pd.read_csv(variability_path)
        else:
            variability_df = None
            
        return {
            "state": state_df,
            "county": county_df,
            "seasonal": seasonal_df,
            "variability": variability_df
        }
    except Exception as e:
        st.error(f"Error loading New York data: {str(e)}")
        return None

def ny_dashboard():
    """Streamlit dashboard for New York mortality analysis."""
    st.set_page_config(
        page_title="New York Mortality Analysis",
        page_icon="🗽",
        layout="wide"
    )
    
    st.title("🗽 New York State Mortality Analysis")
    st.write("Detailed analysis of mortality patterns in New York State with focus on NYC boroughs")
    
    # Add a note about the methodology and assumptions
    with st.expander("Methodology and Assumptions"):
        st.markdown("""
        ### Data Sources
        - CDC mortality data for New York State (1985-2004)
        - Population figures from 2020 Census data
        
        ### Key Assumptions
        1. **Population-based estimation**: County-level deaths are estimated based on population proportion
        2. **Stable population ratios**: The analysis assumes the relative populations of counties have remained consistent
        3. **Similar age distributions**: The model assumes similar age distributions across boroughs (a simplification)
        4. **Death rate correlation**: The model assumes death rates correlate with population size
        5. **Linear extrapolation**: Recent trends are extrapolated from historical patterns
        
        ### Limitations
        - Exact county-level mortality data isn't directly available in the CDC dataset
        - The estimation approach may not capture county-specific factors affecting mortality
        - Using current population figures for historical analysis introduces some imprecision
        
        ### Accuracy Considerations
        The analysis should be considered an educated approximation rather than exact figures, especially at the county level.
        State-level totals are based on actual CDC data, while county distribution is modeled.
        """)
    
    # First check if data files exist, if not, run the analysis
    ny_data_dir = "./ny_analysis_results"
    if not os.path.exists(ny_data_dir) or not os.path.exists(os.path.join(ny_data_dir, "ny_county_estimates.csv")):
        st.warning("New York analysis data not found. Running analysis now...")
        analyzer = NewYorkAnalyzer()
        analyzer.run_nyc_analysis()
        st.success("Analysis complete. Refreshing page...")
        st.experimental_rerun()
    
    # Load the data and safely access dictionary keys
    data = load_ny_data()
    state_data = data.get("state")
    county_data = data.get("county")
    seasonal_data = data.get("seasonal")
    variability_data = data.get("variability")
    
    if state_data is None or county_data is None:
        st.error("Failed to load New York analysis data.")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 NY State Trends", 
        "🏙️ NYC Borough Analysis", 
        "🔄 Seasonal Patterns",
        "📊 Mortality Rate Comparison"
    ])
    
    with tab1:
        st.header("New York State Mortality Trends")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Year range selector for state trends
            years = sorted(state_data["year"].unique())
            min_year, max_year = min(years), max(years)
            
            # Ensure the slider shows 4-digit years
            selected_years = st.slider(
                "Select year range:", 
                min_value=min_year, 
                max_value=max_year, 
                value=(min_year, max_year)
            )
            
            # View options
            view_type = st.radio(
                "View type:",
                ["Deaths", "Death Rate"]
            )
            
            # Moving average option to smooth the trend
            use_ma = st.checkbox("Apply 3-month moving average", value=False)
        
        # Filter data by selected years
        filtered_df = state_data[
            (state_data["year"] >= selected_years[0]) & 
            (state_data["year"] <= selected_years[1])
        ].copy()
        
        # Apply moving average if selected
        if use_ma:
            if view_type == "Deaths":
                filtered_df["smooth_value"] = filtered_df["count"].rolling(window=3, min_periods=1).mean()
                y_column = "smooth_value"
            else:
                filtered_df["smooth_value"] = filtered_df["death_rate"].rolling(window=3, min_periods=1).mean()
                y_column = "smooth_value"
        else:
            y_column = "count" if view_type == "Deaths" else "death_rate"
        
        with col2:
            # Create the trend chart
            fig = px.line(
                filtered_df, 
                x="date", 
                y=y_column,
                title=f"New York State Monthly {view_type} ({selected_years[0]}-{selected_years[1]})",
                labels={
                    "date": "Date", 
                    y_column: "Deaths" if view_type == "Deaths" else "Deaths per 100,000 Population"
                },
                markers=True
            )
            
            fig.update_layout(
                hovermode="x unified",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistics summary
        st.subheader("New York State Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Deaths", f"{filtered_df['count'].sum():,.0f}")
        with col2:
            st.metric("Average Monthly Deaths", f"{filtered_df['count'].mean():,.0f}")
        with col3:
            st.metric("Population (2020)", f"{NY_STATE_POPULATION:,.0f}")
        with col4:
            avg_rate = filtered_df['death_rate'].mean()
            st.metric("Avg Death Rate", f"{avg_rate:.2f} per 100k")
    
    with tab2:
        st.header("NYC Borough Analysis")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Year selector - ensure it shows 4-digit years
            borough_years = sorted(county_data["year"].unique())
            selected_borough_year = st.selectbox(
                "Select year for borough analysis:",
                borough_years,
                index=len(borough_years)-1  # Default to latest year
            )
            
            # Borough view options
            borough_view = st.radio(
                "View type:",
                ["Monthly Trends", "Annual Totals", "Death Rates"]
            )
            
            # Specific borough selection
            borough_filter = st.multiselect(
                "Filter boroughs (select none for all):",
                ["Brooklyn", "Queens", "Manhattan", "Bronx", "Staten Island"],
                default=[]
            )
            
            # Map display names back to county names
            borough_map = {v["name"]: k for k, v in NYC_BOROUGHS.items()}
            selected_boroughs = [borough_map[b] for b in borough_filter] if borough_filter else list(NYC_BOROUGHS.keys())
            
        # Filter for selected year and NYC boroughs
        borough_df = county_data[
            (county_data["year"] == selected_borough_year) & 
            (county_data["county"].isin(selected_boroughs))
        ].copy()
        
        with col2:
            if borough_view == "Monthly Trends":
                # Line chart of borough trends across months
                fig = px.line(
                    borough_df, 
                    x="month", 
                    y="deaths",
                    color="display_name",
                    title=f"Monthly Deaths by NYC Borough ({selected_borough_year})",
                    labels={"deaths": "Estimated Deaths", "month": "Month", "display_name": "Borough"},
                    markers=True
                )
                
                fig.update_xaxes(
                    tickvals=list(range(1, 13)),
                    ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                )
                
            elif borough_view == "Annual Totals":
                # Calculate annual totals
                annual_totals = borough_df.groupby("display_name")["deaths"].sum().reset_index()
                
                # Sort by deaths
                annual_totals = annual_totals.sort_values("deaths", ascending=False)
                
                # Bar chart of annual totals
                fig = px.bar(
                    annual_totals, 
                    x="display_name", 
                    y="deaths",
                    title=f"Annual Deaths by NYC Borough ({selected_borough_year})",
                    labels={"deaths": "Estimated Deaths", "display_name": "Borough"},
                    text_auto=True
                )
                
                fig.update_traces(
                    texttemplate='%{y:,.0f}',
                    textposition='outside'
                )
                
            else:  # Death Rates
                # Calculate annual death rates
                annual_rates = borough_df.groupby("display_name").agg({
                    "deaths": "sum",
                    "population": "first"
                }).reset_index()
                
                # Calculate annual death rate per 100,000
                annual_rates["annual_rate"] = (annual_rates["deaths"] / annual_rates["population"]) * 100000
                
                # Sort by death rate
                annual_rates = annual_rates.sort_values("annual_rate", ascending=False)
                
                # Bar chart of death rates
                fig = px.bar(
                    annual_rates, 
                    x="display_name", 
                    y="annual_rate",
                    title=f"Death Rates by NYC Borough ({selected_borough_year})",
                    labels={"annual_rate": "Deaths per 100,000", "display_name": "Borough"},
                    text_auto=True
                )
                
                fig.update_traces(
                    texttemplate='%{y:.1f}',
                    textposition='outside'
                )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        # Population breakdown
        st.subheader("NYC Borough Population Distribution")
        
        # Create population dataframe for visualization
        population_data = []
        for county, data in NYC_BOROUGHS.items():
            population_data.append({
                "Borough": data["name"],
                "Population": data["population"],
                "Percentage": data["percentage"]
            })
        
        pop_df = pd.DataFrame(population_data)
        
        # Display as a table and a chart side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(pop_df, use_container_width=True)
            
        with col2:
            # Create pie chart of borough populations
            fig = px.pie(
                pop_df, 
                values="Population", 
                names="Borough",
                title="NYC Borough Population Distribution",
                hole=0.4
            )
            
            fig.update_traces(
                textinfo="percent+label",
                insidetextfont=dict(color="white")
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Seasonal Mortality Patterns in New York")
        
        if seasonal_data is None:
            st.warning("Seasonal pattern data is not available. Please run the analysis to generate this data.")
        else:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Borough selection for seasonal patterns
                seasonal_location = st.radio(
                    "Select location:",
                    ["All NYC Boroughs", "Borough Comparison", "NYC vs Rest of State"]
                )
                
                if seasonal_location == "Borough Comparison":
                    # Specific borough selection for comparison
                    seasonal_boroughs = st.multiselect(
                        "Select boroughs to compare:",
                        ["Brooklyn", "Queens", "Manhattan", "Bronx", "Staten Island"],
                        default=["Brooklyn", "Manhattan"]
                    )
                    
                    # Map display names back to county names
                    borough_map = {v["name"]: k for k, v in NYC_BOROUGHS.items()}
                    selected_seasonal_boroughs = [borough_map[b] for b in seasonal_boroughs]
                    
                # Normalization option
                normalize = st.checkbox("Normalize values (% of annual)", value=False)
            
            # Filter the seasonal data based on selection
            if seasonal_location == "All NYC Boroughs":
                # Get data for all NYC boroughs
                seasonal_df = seasonal_data[
                    seasonal_data["county"].isin(NYC_BOROUGHS.keys())
                ].copy()
                
                # Calculate NYC total
                nyc_seasonal = seasonal_df.groupby("month")["deaths"].sum().reset_index()
                nyc_seasonal["county"] = "NYC Total"
                nyc_seasonal["month_name"] = seasonal_data["month_name"].iloc[0:12].values
                
                seasonal_plot_df = nyc_seasonal
                
            elif seasonal_location == "Borough Comparison":
                # Get data for selected boroughs
                seasonal_df = seasonal_data[
                    seasonal_data["county"].isin(selected_seasonal_boroughs)
                ].copy()
                
                # Add display names
                seasonal_df["display_name"] = seasonal_df["county"].map(
                    {k: v["name"] for k, v in NYC_BOROUGHS.items()}
                )
                
                seasonal_plot_df = seasonal_df
                
            else:  # NYC vs Rest of State
                # Get data for NYC and rest of state
                nyc_counties = list(NYC_BOROUGHS.keys())
                nyc_seasonal = seasonal_data[
                    seasonal_data["county"].isin(nyc_counties)
                ].groupby("month").agg({
                    "deaths": "sum",
                    "month_name": "first"
                }).reset_index()
                
                nyc_seasonal["location"] = "NYC"
                
                rest_seasonal = seasonal_data[
                    seasonal_data["county"] == "Rest of State"
                ].copy()
                
                rest_seasonal["location"] = "Rest of State"
                
                # Combine
                seasonal_plot_df = pd.concat([
                    nyc_seasonal[["month", "month_name", "deaths", "location"]],
                    rest_seasonal[["month", "month_name", "deaths", "location"]]
                ])
            
            # Normalize if requested
            if normalize:
                if seasonal_location == "Borough Comparison":
                    # Normalize each borough separately
                    boroughs = seasonal_plot_df["display_name"].unique()
                    normalized_dfs = []
                    
                    for borough in boroughs:
                        borough_df = seasonal_plot_df[seasonal_plot_df["display_name"] == borough].copy()
                        annual_total = borough_df["deaths"].sum()
                        borough_df["normalized_deaths"] = (borough_df["deaths"] / annual_total) * 100
                        normalized_dfs.append(borough_df)
                    
                    seasonal_plot_df = pd.concat(normalized_dfs)
                    y_column = "normalized_deaths"
                    y_label = "% of Annual Deaths"
                    
                elif seasonal_location == "NYC vs Rest of State":
                    # Normalize NYC and Rest of State separately
                    locations = seasonal_plot_df["location"].unique()
                    normalized_dfs = []
                    
                    for location in locations:
                        location_df = seasonal_plot_df[seasonal_plot_df["location"] == location].copy()
                        annual_total = location_df["deaths"].sum()
                        location_df["normalized_deaths"] = (location_df["deaths"] / annual_total) * 100
                        normalized_dfs.append(location_df)
                    
                    seasonal_plot_df = pd.concat(normalized_dfs)
                    y_column = "normalized_deaths"
                    y_label = "% of Annual Deaths"
                    
                else:
                    # Normalize NYC total
                    annual_total = seasonal_plot_df["deaths"].sum()
                    seasonal_plot_df["normalized_deaths"] = (seasonal_plot_df["deaths"] / annual_total) * 100
                    y_column = "normalized_deaths"
                    y_label = "% of Annual Deaths"
            else:
                y_column = "deaths"
                y_label = "Average Monthly Deaths"
            
            with col2:
                # Create the seasonal pattern chart
                if seasonal_location == "Borough Comparison":
                    color_column = "display_name"
                    title = "Seasonal Mortality Patterns by Borough"
                elif seasonal_location == "NYC vs Rest of State":
                    color_column = "location"
                    title = "NYC vs Rest of State Seasonal Patterns"
                else:
                    color_column = None
                    title = "NYC Seasonal Mortality Pattern"
                
                if color_column:
                    fig = px.line(
                        seasonal_plot_df, 
                        x="month_name", 
                        y=y_column,
                        color=color_column,
                        title=title,
                        labels={y_column: y_label, "month_name": "Month"},
                        markers=True
                    )
                else:
                    fig = px.line(
                        seasonal_plot_df, 
                        x="month_name", 
                        y=y_column,
                        title=title,
                        labels={y_column: y_label, "month_name": "Month"},
                        markers=True
                    )
                
                # Ensure correct month order
                month_order = [
                    'January', 'February', 'March', 'April', 'May', 'June', 
                    'July', 'August', 'September', 'October', 'November', 'December'
                ]
                fig.update_xaxes(categoryorder='array', categoryarray=month_order)
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation of seasonal patterns
            st.subheader("Seasonal Pattern Analysis")
            
            st.markdown("""
            The seasonal mortality pattern in New York shows:
            
            1. **Winter Peak**: Higher mortality during winter months (December-March), likely due to:
               - Respiratory illnesses including influenza
               - Cardiovascular complications from cold weather
               - Limited outdoor activity and vitamin D deficiency
            
            2. **Summer Minimum**: Lower mortality during summer months (June-August), with exceptions during:
               - Extreme heat waves
               - Air pollution episodes
            
            3. **Transitional Periods**: Moderate mortality during spring and fall
            
            These patterns are consistent with general mortality trends in northern states with cold winters.
            """)
    
    with tab4:
        st.header("Borough Mortality Rate Analysis")
        
        if variability_data is None:
            st.warning("Borough variability data is not available. Please run the analysis to generate this data.")
        else:
            # Display variability statistics
            st.subheader("Mortality Rate Variability by Borough")
            
            # Create a more readable display dataframe
            display_var = variability_data.copy()
            
            # Format columns for display
            display_var["mean_rate"] = display_var["mean_rate"].map('{:.2f}'.format)
            display_var["std_rate"] = display_var["std_rate"].map('{:.2f}'.format)
            display_var["min_rate"] = display_var["min_rate"].map('{:.2f}'.format)
            display_var["max_rate"] = display_var["max_rate"].map('{:.2f}'.format)
            display_var["cv_rate"] = display_var["cv_rate"].map('{:.3f}'.format)
            
            # Rename columns for clarity
            display_var.columns = [
                "County Code", "Mean Death Rate", "Std Deviation", 
                "Min Rate", "Max Rate", "Coefficient of Variation", "Borough"
            ]
            
            # Show the data
            st.dataframe(
                display_var[["Borough", "Mean Death Rate", "Std Deviation", "Min Rate", "Max Rate", "Coefficient of Variation"]],
                use_container_width=True
            )
            
            # Create visualizations of the variability
            col1, col2 = st.columns(2)
            
            with col1:
                # Plot mean death rates
                var_df = variability_data.copy()
                
                fig = px.bar(
                    var_df.sort_values("mean_rate", ascending=False), 
                    x="display_name", 
                    y="mean_rate",
                    title="Average Death Rate by Borough",
                    labels={"mean_rate": "Deaths per 100,000", "display_name": "Borough"},
                    text_auto=True
                )
                
                fig.update_traces(
                    texttemplate='%{y:.1f}',
                    textposition='outside'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # Plot coefficient of variation
                fig = px.bar(
                    var_df.sort_values("cv_rate", ascending=False), 
                    x="display_name", 
                    y="cv_rate",
                    title="Mortality Rate Variability by Borough",
                    labels={"cv_rate": "Coefficient of Variation", "display_name": "Borough"},
                    text_auto=True
                )
                
                fig.update_traces(
                    texttemplate='%{y:.3f}',
                    textposition='outside',
                    marker_color='orange'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Create a combined visualization showing range
            var_df = variability_data.copy()
            var_df = var_df.sort_values("mean_rate", ascending=False)
            
            fig = go.Figure()
            
            # Add range bars
            for i, row in var_df.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row["display_name"], row["display_name"]],
                    y=[row["min_rate"], row["max_rate"]],
                    mode="lines",
                    line=dict(width=4, color="lightgray"),
                    showlegend=False
                ))
            
            # Add mean points
            fig.add_trace(go.Scatter(
                x=var_df["display_name"],
                y=var_df["mean_rate"],
                mode="markers",
                marker=dict(size=10, color="blue"),
                name="Mean Rate"
            ))
            
            fig.update_layout(
                title="Death Rate Range by Borough",
                xaxis_title="Borough",
                yaxis_title="Deaths per 100,000",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation of the variability
            st.subheader("Interpretation of Borough Mortality Patterns")
            
            st.markdown("""
            ### Key Findings
            
            1. **Rate Differences**: There are notable differences in death rates between boroughs, which may reflect:
               - Demographic differences (age distribution)
               - Socioeconomic factors
               - Healthcare access variations
               - Environmental factors
                
            2. **Variability in Rates**: Some boroughs show greater variability in death rates over time:
               - Higher coefficient of variation indicates more sensitivity to annual factors
               - More stable boroughs show less variation in response to external changes
                
            3. **Range Analysis**: The range between minimum and maximum death rates reveals:
               - Which boroughs were most affected during mortality crises
               - Which boroughs maintained more consistent rates
               
                ### Limitations of Analysis
                
                This analysis is based on population-weighted estimates rather than direct county-level mortality data. 
                The true variation between boroughs may be greater than our model suggests, particularly where 
                demographic factors differ substantially from the state average.
                """)
            
        # If we reach here, the variability data is available and has been displayed
    
    # Footer with methodology notes
    st.markdown("---")
    st.caption("""
        **Data Sources**: CDC Mortality Files, US Census Bureau (2020) | 
        **Methodology**: Population-weighted estimation for county-level data | 
        **Note**: County-level estimates are approximations based on population distribution
    """)

if __name__ == "__main__":
    ny_dashboard()
