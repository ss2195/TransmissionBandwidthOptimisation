import Site
import pandas as pd
from datetime import datetime
import colorsys
import os
import time
import folium

MainNetwork = Site.Network("No File Loaded")


class index:
    """
    Index class is a container for all the backend processing when the four

    input files (PCM file, Coordinate file, Link path file and KPI file) are uploaded.
    """

    def __init__(self):
        """
        Initialize the index class
        """

        self.inp = ['', '', '']  #inp is a list containing objects of uploaded files(PCM, Coordinate, LPF files)
        self.HC = False          #Hard Clean
        self.ULL = False         #Update latitude longitude
        self.st = ""             #Start date
        self.en = ""             #End date
        self.KPI_Files = ""      #File objects of KPI files

    def perform(self, new_format=False):

        """
        :param: takes boolean value based on the button selected
        :return:
        """
        global MainNetwork
        if self.inp[0] != "":
            MainNetwork = Site.Network(self.inp[0])
            MainNetwork.populate()
            MainNetwork.generate_paths(True)
            MainNetwork.read_coordinates(self.inp[1], new_format)
            KPIFL = self.KPI_Files
            if self.HC:

                startt = list(map(int, self.st.split('/')))
                endtt = list(map(int, self.en.split('/')))
                MainNetwork.load_kpi(KPIFL, True, datetime(startt[2], startt[1], startt[0]),
                                     datetime(endtt[2], endtt[1], endtt[0]))
            else:
                MainNetwork.load_kpi(KPIFL)
            if self.inp[2] != "":
                MainNetwork.Load_Link_Data(self.inp[2], self.ULL)
            MainNetwork.propagate()
            MainNetwork.set_occupancy()
        else:
            pass



class analysis:

    def __init__(self, mode=0):
        """
        initialize the attributes of the class
        """
        self.mode = mode
        self.pre_mode = 0
        #self.database = []
        self.limit = 0
        self.varEntry = 1.0
        self.database = ['', '']
        self.tddfdd = 'TDD'
        self.gtp = 0

    def getResults(self):

        """
        Filters out value on th ebasis of threshold provided
        :return:
        """
        ret = []
        head = []
        self.mode = self.pre_mode
        print(self.mode)
        if self.mode == 0:
            val = float(self.varEntry)
            if val < 0.0 or val > 1.0:
                print("value should be between 1 and 0")
                return
            for x in MainNetwork.all_sites:
                print("val",val)
                print("health",MainNetwork.all_sites[x].health)
                if val > MainNetwork.all_sites[x].health > -0.5:
                    print(MainNetwork.all_sites[x].pop)
                    ret.append([x, MainNetwork.all_sites[x].pop, MainNetwork.all_sites[x].health])
            head = ["Site ID", "POP", "Health"]

        elif self.mode == 1:
            val = float(self.varEntry)
            if val < 0 or val > 1.0:
                print("Invalid threshold given")
                return
            for x in MainNetwork.all_sites:
                if val > MainNetwork.all_sites[x].urous_jurool > 0:
                    ret.append([x, MainNetwork.all_sites[x].pop, MainNetwork.all_sites[x].urous_jurool])
            head = ["Site ID", "POP", "Health"]

        elif self.mode == 2:
            val = int(self.varEntry)
            self.limit = val
            ls = MainNetwork.get_faults(val)
            for y in ls:
                x = y[0]
                print(len(x))
                ret.append([x, MainNetwork.all_sites[x].pop, MainNetwork.all_sites[x].max_occ])
            head = ["Site ID", "POP", "Max Data"]

        elif self.mode == 3:
            val = int(self.varEntry)
            self.limit = val
            ls = MainNetwork.get_faults(val, 2, True)
            for y in ls:
                x = y[0]
                ret.append([x, MainNetwork.all_sites[x].pop, MainNetwork.all_sites[x].child_count])
            head = ["Site ID", "POP", "Child Count"]

        elif self.mode == 4:
            val = float(self.varEntry)
            self.limit = val
            for x in MainNetwork.all_sites:
                if MainNetwork.all_sites[x].max_occupancy > val:
                    ret.append([x, MainNetwork.all_sites[x].parent, MainNetwork.all_sites[x].pop,
                                MainNetwork.all_sites[x].max_occupancy])
            head = ["Site ID", "Parent", "POP", "Max Occupancy"]

        else:
            pass

        return self.populateList(ret, head)


    def populateList(self, data, head):
        try:
            df = pd.DataFrame(data)
            df.columns = head
            print(df.head(5))
        except:
            pass
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filepath = os.path.dirname(os.path.abspath(__file__))
        print(filepath)
        filename = filepath+"/inputs/data"+timestr+".csv"
        df.to_csv(filename)
        with open(filename) as f:
            s = f.read() + '\n'  # add trailing new line character
            try:
                print(s)
            except:
                pass
        os.remove(filename)
        return s

    def setMode(self, mode):
        self.pre_mode = mode

    def get_graph(self):

        if self.mode == 0:
            print("graph")
            site_id = self.database[0]

            if self.tddfdd == 'TDD':
                fig = MainNetwork.all_sites[site_id].plot(self.gtp, False, -1)
                return fig
            else:
                fig = MainNetwork.all_sites[site_id].plot(self.gtp, True, -1)
                return fig
        elif self.mode == 1:
            print("graph")
            if self.gtp == 2:
                site_id = self.database[0]
                fig = MainNetwork.all_sites[site_id].plot_downstream(False, -1)
            else:
                if self.database[1] != "self":
                    print("Aggregates for columns other that PCDP trans are available only at the POP level. Displaying graph for corresponding POP")
                    sel = self.database[1]
                else:
                    sel = self.database[0]
                print("Getting graph for", sel)
                strt = MainNetwork.all_sites[sel].downstream_kpi[0, 0]
                endt = MainNetwork.all_sites[sel].downstream_kpi[-1, 0]
                fig = MainNetwork.Plot_POP_Aggr(strt, endt, sel, self.gtp, -1, 2)
            return fig
        elif self.mode == 2:
            print("graph")
            site_id = self.database[0]
            fig = MainNetwork.all_sites[site_id].plot(2, (self.tddfdd == 'FDD'), self.limit)
            return fig
        elif self.mode == 3:
            site_id = self.database[0]
            fig = MainNetwork.all_sites[site_id].plot_downstream(False, self.limit)
            return fig
        elif self.mode == 4:
            print("graph")
            site_id = self.database[0]
            fig = MainNetwork.all_sites[site_id].plot_downstream(True, self.limit)
            return fig

    def plot_folium(self):
        POP_ID = MainNetwork.all_sites[self.database[0]].pop
        if POP_ID == "self":
            POP_ID = self.database[0]

        def color(val):
            if val == -1:
                return "#000000"
            r, g, b = colorsys.hsv_to_rgb(val / 3.0, 1, 1)
            h_out = hex(int(r * 256 * 256 * 256 + g * 256 * 256 + b * 256))
            return "#" + "0" * (6 - len(h_out[2:])) + h_out[2:]

        if POP_ID not in MainNetwork.all_sites or MainNetwork.all_sites[POP_ID].lat in {0, 500}:
            print("Cannot plot this POP")
            return 'Error_in_map.html'
        coord = (MainNetwork.all_sites[POP_ID].lat, MainNetwork.all_sites[POP_ID].long)
        mp = folium.Map(coord, tiles="OpenStreetMap", width='100%', height='100%', zoom_start=14)
        pts = []
        for x in MainNetwork.all_sites[POP_ID].descendants:
            if MainNetwork.all_sites[x].lat not in {None, 0, 500} and MainNetwork.all_sites[MainNetwork.all_sites[x].parent].lat not in {None, 0, 500}:
                pts.append(
                    (MainNetwork.all_sites[x].lat, MainNetwork.all_sites[x].long, MainNetwork.all_sites[x].health,
                     MainNetwork.all_sites[MainNetwork.all_sites[x].parent].lat,
                     MainNetwork.all_sites[MainNetwork.all_sites[x].parent].long, x))

        for p in pts:
            folium.Marker(location=(p[0], p[1]), icon=folium.Icon(color=color(p[2])),
                          tooltip="SiteID: " + p[-1]).add_to(mp)
            folium.Marker(location=(p[3], p[4]), icon=folium.Icon(color=color(p[2])),
                          tooltip="SiteID: " + p[-1]).add_to(mp)
            folium.Circle((p[0], p[1]), 20, color='blue', fill=True).add_to(mp)
            folium.PolyLine(locations=((p[0], p[1]), (p[3], p[4])), color='black').add_to(mp)


        folium.Marker(location=(MainNetwork.all_sites[POP_ID].lat, MainNetwork.all_sites[POP_ID].long),
                      icon=folium.Icon(color='green'), tooltip="SiteID: "+POP_ID).add_to(mp)

        filepath = os.path.dirname(os.path.abspath(__file__)) + "/templates/"
        filename = "map" + time.strftime("%Y%m%d-%H%M%S") + ".html"
        mp.save(filepath + filename)  # map gets saved as html file inside templates folder
        return filename

