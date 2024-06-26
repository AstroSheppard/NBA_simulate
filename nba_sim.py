import pandas as pd
import numpy as np
import csv
import requests
import sys
import random
from datetime import date, timedelta, datetime
from urllib2 import urlopen
import bs4
from bs4 import BeautifulSoup as bs
import matplotlib.pyplot as plt
import schedule
import time
from tqdm import trange

def get_prob(delta):
    return 1./(1.0+10**(-1*delta/400.))

def get_standings(past):
    """Return the current NBA standings"""
    own=np.asarray([['GSW', 'LAC', 'MEM', 'PHO','ORL']
                    ,['POR','HOU','TOR','DET','NYK']
                    ,['CLE','MIN','UTA','IND','BRK']
                    ,['DAL','MIL','SAS','SAC','NOP']
                    ,['OKC','DEN','ATL','PHI','MIA']
                    ,['BOS','CHO','CHI','WAS','LAL']])
    nWins=[]
    for team_list in own:       
        sum=0
        for name in team_list:
            home=past[past['team1']==name]['T1_win']
            away=1-past[past['team2']==name]['T1_win']
            sum=sum+(home.sum()+away.sum())
        nWins.append(sum)
    return nWins
    

def do_sim(schedule, standings):

    # Dumb, annoying way to show who owns who
    own=np.asarray([['GSW', 'LAC', 'MEM', 'PHO','ORL']
                    ,['POR','HOU','TOR','DET','NYK']
                    ,['CLE','MIN','UTA','IND','BRK']
                    ,['DAL','MIL','SAS','SAC','NOP']
                    ,['OKC','DEN','ATL','PHI','MIA']
                    ,['BOS','CHO','CHI','WAS','LAL']])


    test=np.random.rand(len(schedule))
    schedule['T1_Win']=test<schedule['prob1']
    schedule['T2_Win']=1-schedule['T1_Win']
    
    # We now have the result of each game. Must find total wins for each
    # team, then for each owner
    results=pd.DataFrame()
    results['Owner']=['Myra', 'Keegan', 'Kyle',
                      'Kunal', 'Blake', 'Mike']
    nWins=[]
    for i,team_list in enumerate(own):
        sum=standings[i]
        for name in team_list:
            home=schedule.loc[schedule.loc[:,'team1']==name,'T1_Win']
            away=schedule.loc[schedule.loc[:,'team2']==name,'T2_Win']
            sum=sum+np.sum(home.values)+ np.sum(away.values)
        nWins.append(sum)
    results['Wins']=nWins
    return results.sort_values(by='Wins', ascending=False)

def get_hists(df, player, direct):
    """ Using simulation results, plot and save the distribution
    of possible places and games won """
    # First get lists of point, rank and money for the player
    wins=df[df['Owner']==player].loc[:,'Wins'].values
    rank=df[df['Owner']==player].loc[:,'Rank'].values
    
     
    # Determine and save money results
    pwin=[1 if item==1 else 0 for item in rank]
    p2=[1 if item==2 else 0 for item in rank]
    percent_win=np.round(np.sum(pwin)/float(len(rank))*100)
    percent_2nd=np.round(np.sum(p2)/float(len(rank))*100)
    percent_cash=percent_win+percent_2nd
    avg=np.mean(wins)
    std=np.std(wins)
    cash_info=[percent_win, percent_2nd, percent_cash, avg, std]

    # Make the figures: 2 histograms on 1 page
    fig, axes = plt.subplots(nrows=2, ncols=1)
    ax0, ax1 = axes.flatten()
    weights = np.ones_like(wins)/float(len(wins)) # Make height = probability
    ax1.hist(wins,bins=12, weights=weights)
    ax1.set_title('Wins Distribution', fontsize=10)
    ax1.set_xlabel('Wins')
    ax1.set_ylabel('Frequency')

    weights = np.ones_like(rank)/float(len(rank))
    ax0.hist(rank,bins=max(rank), weights=weights)
    ax0.set_title('Rank Distribution', fontsize=10)
    ax0.set_xlabel('Rank')     
    ax0.set_ylabel('Frequency')
    
    fig.tight_layout()
    plt.text(0.3, 0.9, "Winning freq: " + str(percent_win)+'%',
             horizontalalignment='center',
             verticalalignment='center', fontsize=8,
             transform = ax0.transAxes)
    plt.text(0.3, 0.78,"Cashing freq: " + str(percent_cash) + "%",
             horizontalalignment='center',
             verticalalignment='center', fontsize=8,
             transform = ax0.transAxes)
    fig.savefig(direct+player+'_chances.png')
   # plt.show()

    return cash_info





###########################################################
if __name__ == "__main__":

    CSV_URL='https://projects.fivethirtyeight.com/nba-model/nba_elo.csv'

    with requests.Session() as s:
        download = s.get(CSV_URL)
        decoded_content = download.content.decode('utf-8')

        # Convert to dataframe

        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        my_list = list(cr)
        head, my_list=my_list[0], my_list[1:]
        df=pd.DataFrame(my_list, columns=head)

    #Extract current date predictions

    df['dif']=[(datetime.strptime(date,'%Y-%m-%d')-datetime.today()).days
               for date in df['date'].values]

    future=df[df['dif'] > -2].drop(columns=['season','neutral','playoff',
                                            'elo1_pre' ,'dif', 'carmelo2_post'
                                            ,'carmelo_prob1','carmelo_prob2'
                                            ,'carmelo1_post','elo1_post'
                                            ,'elo2_post', 'elo2_pre','elo_prob2'
                                            ,'elo_prob1', 'score1','score2'])

    past=df[(df['dif'] <= -2) & (
        df['season'] == '2018')].drop(columns=['season','neutral','playoff'
                                               ,'elo1_pre' ,'dif'
                                               ,'carmelo2_post','carmelo_prob1'
                                               ,'carmelo_prob2','carmelo1_post'
                                               ,'elo1_post','elo2_post'
                                               ,'elo2_pre','elo_prob2'
                                               ,'elo_prob1'])
    past['T1_win'] = [float(past['score1'].values[i])>
                      float(past['score2'].values[i])
                      for i in range(len(past))]

    probs=[get_prob(float(future['carmelo1_pre'].values[i])
                    - float(future['carmelo2_pre'].values[i])
                    + 100.) for i in range(len(future))]
    future['prob1']=probs


    # Simulate season, then import current rankings and add current wins
    nSim=5000
    rank=range(1,7)
    results=pd.DataFrame()
    standings=get_standings(past)
    for i in trange(nSim):
        run=do_sim(future, standings)
        run['Rank']=rank
        results=results.append(run)
        #print i


    results=results.sort_values(by='Owner')
    players=results['Owner'].drop_duplicates()

    dirr='./player_chances/'
    to_df=[]
    for play in players:
        cash=[play]+get_hists(results, play, dirr)
        print play
        to_df.append(cash)


    output=pd.DataFrame(to_df, columns=['Player','Freq Win', 'Freq 2nd'
                                        ,'Freq Cash','Average wins','Std Dev'])
    output=output[:].apply(pd.to_numeric, errors='ignore')
    output=output.sort_values('Freq Cash', ascending=False)
    output.iloc[:,1:4]=output.iloc[:,1:4]/100.
    output.to_csv(dirr+'NBA_sim.csv')


