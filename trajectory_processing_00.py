## PRC Data Challenge - M3 Systems Team
## Author: Jose Gonzalez - github: jraniero
## Date: 2024-10-15
## License: GPLv3
## Description: 
##  This file pre-process the .parquet files with the challenge trajectory data
##  FOllowing data is extracted from trajectory
##    - Estimation of maximum flight level: using the statistical mode, median and maximum
##    - Estimation of first climb plateau: First stable altitude maintained
##    - Estimation of average vertical rate until first plateau
##    - Estimation of windspeed and airspeed at take-off, with magnitude and direction
##    - Estimaton of windspeed and airspeed at first plateau, with magnitude and direction
## This code has been written with speed efficiency in mind, given we had no big data infrastructure available
## Therefore, we are not using the traffic library and we are using heuristics methods for extracting
## the trajectory data.

import os
import pandas as pd
import logging
import numpy as np

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


source_folder = os.getenv("SOURCE_FOLDER")
destination_folder=os.getenv("DESTINATION_FOLDER")


challenge_file = source_folder+"/"+os.getenv("CHALLENGE_FILE")
submission_file =source_folder+"/"+os.getenv("SUBMISSION_FILE")

df_metadata = pd.concat([pd.read_csv(challenge_file),pd.read_csv(submission_file)])

df_metadata["aobt_dt"]=pd.to_datetime(df_metadata["actual_offblock_time"])
df_metadata["taxi_out_dt"]=pd.to_timedelta(df_metadata["taxiout_time"],unit='min')
df_metadata["atot"]=df_metadata["aobt_dt"]+df_metadata["taxi_out_dt"]

# SOURCE FOLDER MUST BE THE FOLDER WITH THE CLEANED ALTITUDE DATA, MADE BY THE CELL ABOVE

df_max_fl_all_array = []

df_prev=None
df_cur=None

def modeMax(x):
    return pd.Series.mode(x).max()

filter=[("flight_id","in",df_metadata["flight_id"].unique())]

print(df_metadata["flight_id"].unique())

for filename in os.listdir(source_folder):
    if filename.endswith('.parquet'):
        logging.info(filename)
        file_path = os.path.join(source_folder, filename)
        logging.info(filename+" read START")
        df = pd.read_parquet(file_path,columns=["timestamp","flight_id","altitude","vertical_rate","u_component_of_wind","v_component_of_wind","track","groundspeed"],filters=filter)        
        logging.info(filename+" read END")
        df=df.merge(df_metadata[["flight_id","atot"]],on="flight_id")
        df["timestamp_dt"]=pd.to_datetime(df["timestamp"])
        df=df[df["timestamp_dt"]>=df["atot"]]

        if True:
            logging.info("Wind speed calculation START")
                                    # Wind speed
            df['wind_speed'] = np.sqrt(df['u_component_of_wind']**2 + df['v_component_of_wind']**2)

            df['wind_direction'] = (np.pi/2)-np.arctan2(df['u_component_of_wind'], df['v_component_of_wind'])

            df['airspeed'] = np.sqrt(
                df['groundspeed']**2 +
                df['wind_speed']**2 -
                2 * df['groundspeed'] * df['wind_speed'] * np.cos(((np.pi*df['track']/180)) - df['wind_direction'])
            )

            df['airspeed_angle'] = ((np.pi*df['track']/180) + np.arcsin(
                (df['wind_speed'] * np.sin(df['track'] - df['wind_direction'])) / df['airspeed']
            ))*(180/np.pi)

            df['wind_direction']=df['wind_direction']*(180/np.pi)
            logging.info("Wind speed calculation END")

        df["altitude_prev"]=df.groupby("flight_id")["altitude"].shift(1)
        

        logging.info("Max altitude START")
        df_max_altitude = df.groupby('flight_id').agg(fl_mode=('altitude',modeMax),fl_max=('altitude','max'),fl_median=('altitude',pd.Series.median))
        
        logging.info("Plateau detection START")
        df_first_plateau=df[(df["timestamp_dt"]>=(df["atot"]+pd.Timedelta(10,unit='min'))) & ((df["vertical_rate"]==0) | (df["altitude"]<=(df["altitude_prev"]+200)))].sort_values(by=["flight_id","timestamp_dt"],ascending=True).groupby("flight_id").first().reset_index()
        logging.info("Plateau detection END")

        df_first_plateau.rename(columns={"timestamp_dt":"plateau_timestamp_dt",
                                         "altitude":"plateau_altitude",
                                         "wind_speed":"plateau_wind_speed",
                                         "wind_direction":"plateau_wind_direction",
                                         "airspeed":"plateau_airspeed",
                                         "airspeed_angle":"plateau_airspeed_angle"
                                         },inplace=True)
        df_first_plateau["plateau_climb_duration_min"]=(df_first_plateau["plateau_timestamp_dt"]-df_first_plateau["atot"]).dt.total_seconds()/60
        
        
        df_max_altitude_reached=df.merge(df_max_altitude,on="flight_id")
        df_fl_ground=df.groupby('flight_id').agg(fl_ground=('altitude',"min"),
        ground_wind_speed=('wind_speed',"first"),
        ground_wind_direction=('wind_direction',"first"),
        ground_airspeed=('airspeed',"first"),
        ground_airspeed_angle=('airspeed_angle',"first")
        ).reset_index()

 
        df_max_altitude_reached=df_max_altitude_reached[df_max_altitude_reached["altitude"]>=(0.7*df_max_altitude_reached["fl_mode"])]
        df_max_altitude_reached=df_max_altitude_reached[(df_max_altitude_reached["vertical_rate"]<1000) & (df_max_altitude_reached["vertical_rate"]>-1000)]
        df_climb_rate=df_max_altitude_reached[df_max_altitude_reached["vertical_rate"]>0].groupby("flight_id").agg(vertical_rate_mode=('vertical_rate',modeMax),vertical_rate_max=('vertical_rate','max'),vertical_rate_mean=('vertical_rate','mean')).reset_index()
        df_max_altitude_reached=df_max_altitude_reached.groupby("flight_id").first().reset_index()
        
        df_max_altitude_reached.rename(columns={'timestamp_dt':'max_fl_timestamp_dt'},inplace=True)
        df_max_altitude_reached=df_max_altitude_reached.merge(df_climb_rate[["vertical_rate_mode","vertical_rate_max","flight_id"]],on="flight_id",how="left") 
        df_max_altitude_reached["climb_duration_min"]=(df_max_altitude_reached["max_fl_timestamp_dt"]-df_max_altitude_reached["atot"]).dt.total_seconds()/60
        
        df_max_altitude_reached=df_max_altitude_reached.merge(df_fl_ground,on="flight_id")
        df_max_altitude_reached=df_max_altitude_reached.merge(df_first_plateau[["flight_id",
        "plateau_timestamp_dt",
        "plateau_altitude",
        "plateau_climb_duration_min",
        "plateau_wind_speed",
        "plateau_wind_direction",
        "plateau_airspeed",
        "plateau_airspeed_angle"

        ]],on="flight_id",how="left")
        
        df_max_altitude_reached["fl_max_climb_rate_avg"]=(df_max_altitude_reached["fl_median"]-df_max_altitude_reached["fl_ground"])/df_max_altitude_reached["climb_duration_min"]
        df_max_altitude_reached["plateau_climb_rate_avg"]=(df_max_altitude_reached["plateau_altitude"]-df_max_altitude_reached["fl_ground"])/df_max_altitude_reached["plateau_climb_duration_min"]
                
        df_max_altitude=df_max_altitude.merge(df_max_altitude_reached[["flight_id","timestamp","vertical_rate_mode",                                                                       
                                                                       "ground_wind_speed",
                                                                       "ground_wind_direction",
                                                                       "ground_airspeed",
                                                                       "ground_airspeed_angle",
                                                                       "plateau_climb_duration_min",
                                                                       "plateau_climb_rate_avg",
                                                                       "plateau_altitude",
                                                                       "plateau_climb_duration_min",
                                                                       "plateau_wind_speed",
                                                                       "plateau_wind_direction",
                                                                       "plateau_airspeed",
                                                                       "plateau_airspeed_angle"
                                                                       ]],on="flight_id",how="left")

        df_max_altitude["vertical_rate_mode"]=df_climb_rate["vertical_rate_mode"].max()
        
        df_max_altitude.rename(columns={"timestamp":"Mode_level_timestamp"},inplace=True)
        
        df_max_fl_all_array.append(df_max_altitude)
        logging.info("Max altitude END")
      
        


df_max_fl_all=pd.concat(df_max_fl_all_array)
df_max_fl_all.to_csv(destination_folder+"/"+os.getenv("TRAJECTORY_PREPROCESSING_00_FILE"),index=False)
