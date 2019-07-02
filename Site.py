# coding=utf-8
import csv
#import networkx
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from math import tanh
from functools import reduce
import io


class Site(object):
    """
    Site class to be used within the Network class. Currently acts as a container.
    Author:Sankalan Pal Chowdhury(For Nokia Networks Pvt. Ltd)
    For any queries, please contact Sankalan.story@gmail.com
    """

    def __init__(self, site_id, pop=None, parent=None, verified=False):
        """
        Initialise a new site
        :param site_id: The site id of the current site
        :param pop: The pop to which the current site is connected
        :param parent: The site/POP directly upstream to the current site
        :param verified: True if this site was encountered in the file, false if created for supporting descendants
        """
        # make sure that path terminates in the appropriate pop
        self.site_id = site_id
        self.parent = parent
        self.pop = pop
        self.children = set()
        self.descendants = set()  # applicable only for pops
        self.verified = verified
        self.lat = 0
        self.long = 0
        self.tdd_kpi = []
        self.fdd_kpi = []
        self.tdd_kpi_np = None
        self.fdd_kpi_np = None
        self.fdd_kpi_var = -1.0
        self.tdd_kpi_var = -1.0
        self.uplink = -1.0
        self.kpi_propagated = False
        self.downstream_kpi = None
        self.max_occupancy = -1
        self.kpi_count = -1
        self.urous_jurool = -1
        self.child_count = 0
        self.new_format = False

    @staticmethod
    def __check_data(data, hard_clean=False, start_time=None, end_time=None, threshold=3, reject_low=False,
                     Minimum_entries=0.85):
        # Check to make sure that the data is fine. Fill in holes and replace outlier data with second order mean.
        # Fail in case of too many outliers.
        if not hard_clean:
            start_time = data[0, 0]
            end_time = data[-1, 0]
        delta = end_time - start_time
        expected_Entries = delta.days * 24 + delta.seconds // 3600
        # Blank values were replaced by -1 by our code. If we count numbers above -0.5, we get an idea of how many data
        # points were missing without having to worry about precision issues. If this is too high, we better inform this
        if not hard_clean and (
                expected_Entries == 0 or (data[:, 3] > -0.5).sum() / (expected_Entries * 1.0) < Minimum_entries):
            return data, False
        if hard_clean:
            # step 1: ensure all date entries are present:
            if data.shape[0] < expected_Entries:
                delta = timedelta(hours=1)
                # print "Filling in entries, only", data.shape[0], "present of expected", expected_Entries
                for i in range(expected_Entries):
                    tm = start_time + i * delta
                    if data.shape[0] <= i or data[i, 0] > tm:
                        # insert now
                        data = np.insert(data, i, [tm, 0, 0, -1, 0], axis=0)
                # print data.shape[0]
            # step 2: Replace clear noise entries with -1:
            std = np.std(data[data[:, 3] > -0.5, 3])
            mn = np.mean(data[data[:, 3] > -0.5, 3])
            for y in range(expected_Entries):
                if data[y, 3] > mn + threshold * std or (
                        reject_low and data[y, 3] > -0.5 and data[y, 3] < mn - threshold * std):
                    data[y, 3] = -1
            # step 3: Replace all -1's with sensible values
            valid_indices = np.where(data[:, 3] > -0.5)[0]
            if len(valid_indices) > 2:
                for i in range(expected_Entries):
                    if data[i, 3] < 0:
                        bef = valid_indices.searchsorted(i)
                        if bef == 0:
                            data[i, 3] = data[valid_indices[0], 3]
                        elif bef == len(valid_indices):
                            data[i, 3] = data[valid_indices[bef - 1], 3]
                        else:
                            j = bef
                            k = bef - 1
                            data[i, 3] = ((data[valid_indices[j], 3] - data[valid_indices[k], 3]) * i - valid_indices[
                                k] *
                                          data[valid_indices[j], 3] + valid_indices[j] * data[valid_indices[k], 3]) / (
                                                 valid_indices[j] - valid_indices[k])
                return data, True
            else:
                print("Need atleast 2 entries to interpolate.\n", valid_indices)
                return data, False
        else:
            # Check for entries that are clearly noise
            std = np.std(data[data[:, 3] > -0.5, 3])
            mn = np.mean(data[data[:, 3] > -0.5, 3])
            if max(data[:, 3]) > mn + threshold * std or (reject_low and min(data[:, 3]) < mn - threshold * std):
                # handle first and last entries separately
                if data[0, 3] > mn + threshold * std or (reject_low and data[0, 3] < mn - threshold * std):
                    data[0, 3] = 2 * data[1, 3] - data[2, 3]
                if data[-1, 3] > mn + threshold * std or (reject_low and data[-1, 3] < mn - threshold * std):
                    data[-1, 3] = 2 * data[-2, 3] - data[-3, 3]
                for y in range(1, data.shape[0] - 1):
                    if data[y, 3] > mn + threshold * std or (reject_low and data[y, 3] < mn - threshold * std):
                        data[y, 3] = (data[y - 1, 3] + data[y + 1, 3]) / 2.0
            return data, True

    @property
    def health(self):
        """
        Health is a single number that would encompass how well the site is performing. a more or less constant KPI site
        will have very low health. If both tdd and fdd technologies are available at the site, the lower one will be
        reported. If none of these are present, -1 will be returned.

        Example:
            X=pretty_load(...)
            print X.all_sites["..."].health
        :return: -1 for no data or a value between 0 and 1
        """
        if self.fdd_kpi_var < 0 and self.tdd_kpi_var < 0:
            return -1
        elif self.fdd_kpi_var < 0:
            return max(0.0, tanh(self.tdd_kpi_var * 2.0 - 0.1))
        elif self.tdd_kpi_var < 0:
            return max(0.0, tanh(self.fdd_kpi_var * 2.0 - 0.1))
        else:
            return min(max(0.0, tanh(self.fdd_kpi_var * 2.0 - 0.1)), max(0.0, tanh(self.tdd_kpi_var * 2.0 - 0.1)))

    def add(self, row, type_TDD):
        """
        Add an KPI entry to the site. TDD/FDD infered based on site id.
        :param row: Data in prescribed format as per sample
        :return: Nothing
        """
        if (type_TDD is None and row[1][0] == "8") or type_TDD:
            self.tdd_kpi.append([row[0]] + row[2:])
        else:
            self.fdd_kpi.append([row[0]] + row[2:])

    def initiate(self, row, type_TDD):
        """
        Initiates a set of add operation. Cleans up any previous entries, so avoid calling if appending is intended.
        :param row: First entry. Note that this is already added by this function and a repeat call is not required
        :return: Nothing
        """
        if (type_TDD is None and row[1][0] == "8") or type_TDD:
            self.tdd_kpi = []
        else:
            self.fdd_kpi = []
        self.add(row, type_TDD)

    def cleanup(self, hard_clean=False, start_date=None, end_date=None):
        """
        Does the cleanup after a set of add operations.
        :param hard_clean: If set to true, all missing data will be filled in by first order interpolation
        :param start_date: Required for hard clean. Earliest time for which data is provided.
        :param end_date:   Required for hard clean. Latest time for which data is provided.
        :return: Nothing
        """
        if len(self.tdd_kpi) > 0:
            self.tdd_kpi_np, success = Site.__check_data(np.array(sorted(self.tdd_kpi)), hard_clean, start_date,
                                                         end_date)
            if not success:
                print(self.site_id, "may not have good tdd data. One reason for this could be too many missing entries")
            if not np.mean(self.tdd_kpi_np[self.tdd_kpi_np[:, 3] > -0.5, 3]) == 0:
                self.tdd_kpi_var = np.std(self.tdd_kpi_np[self.tdd_kpi_np[:, 3] > -0.5, 3]) / np.mean(
                    self.tdd_kpi_np[self.tdd_kpi_np[:, 3] > -0.5, 3])
            else:
                print(self.site_id, "has zero mean on tdd values. Correct this or delete this.")
        if len(self.fdd_kpi) > 0:
            self.fdd_kpi_np, success = self.__check_data(np.array(sorted(self.fdd_kpi)), hard_clean, start_date,
                                                         end_date)
            if not success:
                print(self.site_id, "may not have good fdd data. One reason for this could be too many missing entries")
            if not np.mean(self.fdd_kpi_np[:, 3]) == 0:
                self.fdd_kpi_var = np.std(self.fdd_kpi_np[self.fdd_kpi_np[:, 3] > -0.5, 3]) / np.mean(
                    self.fdd_kpi_np[self.fdd_kpi_np[:, 3] > -0.5, 3])
            else:
                print(self.site_id, "has zero mean on fdd values. Correct this or delete this.")

    @property
    def max_occ(self):
        if self.tdd_kpi_np is not None:
            max1 = max(self.tdd_kpi_np[:, 3])
        else:
            max1 = 0
        if self.fdd_kpi_np is not None:
            max2 = max(self.fdd_kpi_np[:, 3])
        else:
            max2 = 0
        return max(max1, max2)

    def plot(self, c_id, fdd=True, limit=150000):
        """
        Creates a plot for the KPI data for the site.
        :param c_id: ID of data to plot. 0 for S1, 1 for X2, 2 for PDCP Data Rate, 3 for Max Act UEs
        :param fdd: Set to false if plot for FDD is wanted
        :param limit: Draws a red line at this value. Set ta a negative value to ignore
        :return: pyplot.figure containing the desired graph
        """
        plt.rcParams['font.size'] = 9
        fig = plt.figure(figsize=(8, 6), dpi=100)
        plt.xlabel("Time")
        if fdd and self.fdd_kpi_np is not None:
            plt.plot(self.fdd_kpi_np[:, 0], self.fdd_kpi_np[:, (c_id + 1)])
            fig.suptitle("FDD-90" + self.padded_id())
        elif not fdd and self.tdd_kpi_np is not None:
            plt.plot(self.tdd_kpi_np[:, 0], self.tdd_kpi_np[:, (c_id + 1)])
            fig.suptitle("TDD-80" + self.padded_id())
        plt.ylabel(
            ["S1 interface SCTP successful transmission ratio(%)", "X2 interface SCTP successful transmission ratio",
             "PDCP_DATA_RATE_MAX_DL (M8012C25)", "Maximum Active UEs with data in the buffer per cell DL"][c_id])
        if c_id == 2 and limit > 0:
            plt.axhline(limit, color="red")
        if c_id == 0:
            plt.ylim(ymax=100.5)
        plt.gcf().autofmt_xdate()
        #plt.shaow()
        return fig

    def plot_downstream(self, child_mode=True, limit=150):
        if self.kpi_propagated:
            plt.rcParams['font.size'] = 9
            fig = plt.figure(figsize=(8, 6), dpi=100)
        fig.suptitle(self.site_id)
        plt.xlabel("Time")
        plt.ylabel("Downstream KPI(KB)")
        plt.plot(self.downstream_kpi[:, 0], self.downstream_kpi[:, 1])
        if self.uplink > 0 and child_mode:
            plt.axhline(self.uplink * 1000, color="red")
            plt.axhline(self.uplink * limit * 1000, color="yellow")
        if child_mode == False and limit > 0:
            plt.axhline(self.child_count * limit, color="red")
        #plt.show()
        return fig

    def padded_id(self):
        '''
        Adds trailing zeroes to make site_id 4 character long. Non-sensical for Alphanumeric IDs
        :return: padded site_id
        '''
        pad = "0000"
        if len(self.site_id) < 5:
            return pad[:(4 - len(self.site_id))] + self.site_id
        else:
            return self.site_id

    def data_between(self, start_time, end_time, fdd=True):
        """
        Retrieve slice of data between given dates. Beneficial for aligning data for addition.
        :param start_time: datetime.datetime object reflecting start time for slice(inclusive)
        :param end_time: datetime.datetime object reflectimg end time for slice(exclusive)
        :param fdd: True for fdd data, false for TDD data.
        :return: numpy array containing all data between indicated periods. 4 Colums having timestanp, S1, X2 and PDCP
        """
        if start_time < end_time:
            start_index = 0
            if fdd and self.fdd_kpi_np is not None:
                out = self.fdd_kpi_np[:, 0].tolist() + [datetime.max]
                while out[start_index] < start_time:
                    start_index += 1
                end_index = start_index
                while out[end_index] < end_time:
                    end_index += 1
                return self.fdd_kpi_np[start_index:end_index]
            elif not fdd and self.tdd_kpi_np is not None:
                out = self.tdd_kpi_np[:, 0].tolist() + [datetime.max]
                while out[start_index] < start_time:
                    start_index += 1
                end_index = start_index
                while out[end_index] < end_time:
                    end_index += 1
                return self.tdd_kpi_np[start_index:end_index]
        else:
            return np.array([])

    def propagate(self, dict, tp=2, depth=10):
        """
        Propagate KPIs from child to POP to find out load on each link. For this to work correctly, data should be
        either noise-free of hard cleaned.
        :param tp: The Type of data required, 0 for fdd,1 for tdd, 2 for combined.
        :param dict: Dictionary of all sites
        :param depth: Maximum number of recursive calls allowed at this point
        :return:
        """
        if depth == 0:
            print("Please check architecture around", self.site_id, "as there seems to be a loop there. Force terminated")
            return
        all_kpis = []
        child_count = 0
        for K in self.children:
            if K != self.site_id and dict[K].kpi_propagated == False:
                dict[K].propagate(dict, tp, depth - 1)
                if dict[K].kpi_propagated:
                    all_kpis.append(dict[K].downstream_kpi)
                    child_count += dict[K].child_count

        if self.tdd_kpi_np is not None and tp > 0:
            all_kpis.append(self.tdd_kpi_np[:, [0, 3]])
            child_count += 1
        if self.fdd_kpi_np is not None and tp != 1:
            all_kpis.append(self.fdd_kpi_np[:, [0, 3]])
            child_count += 1
        if len(all_kpis) > 0:
            # check that all the data is aligned
            right_size = max([p.shape[0] for p in all_kpis])
            good_kpis = [p for p in all_kpis if p.shape[0] == right_size]
            if len(good_kpis) < len(all_kpis):
                try:
                    print("Some kpis were dropped in", self.site_id)
                except:
                    pass
            self.downstream_kpi = reduce(lambda x, y: np.column_stack((x[:, 0], y[:, 1] + x[:, 1])), good_kpis)
            self.kpi_propagated = True
            self.urous_jurool = tanh(np.std(self.downstream_kpi[:, 1]) * 4.0 / np.mean(self.downstream_kpi[:, 1]))
            self.child_count = child_count

    def check_limit(self, limit=150000, tp=2, POP=False):
        """
        Check if the site is exceeding the limit
        :param limit: The limit...Simple, right
        :param tp: The Type of data required, 0 for fdd,1 for tdd, 2 for combined.
        :return: True if there is an issue.
        """
        if not POP:
            if tp == 0 or tp == 2 and self.fdd_kpi_np is not None and max(self.fdd_kpi_np[:, 3]) > limit:
                return True
            if tp == 1 or tp == 2 and self.tdd_kpi_np is not None and max(self.tdd_kpi_np[:, 3]) > limit:
                return True
            return False
        else:
            return self.kpi_propagated and (max(self.downstream_kpi[:, 1]) > (limit * self.child_count))


class Network:
    """
    Master class for drawing all networks. Currently supports reading from csv files(Easily convertible from xlsx files)
    and rendering networks for independent pops.
    Example:
         import Site
         X=Site.Network("Howise PCM.csv")
         X.populate()
         X.generate_paths()
         X.render("6462")

    Dependencies:   Matplotlib
                    NetworkX
                    Numpy
                    Csv

    Author: Sankalan Pal Chowdhury(For Nokia Networks Pvt. Ltd)
    For any queries, please contact Sankalan.story@gmail.com
    """

    def __init__(self, filename):
        '''
        Initialise the network.
        :param filename: Input file name object.
        '''
        self.PIM = []
        self.SIM = []
        self.MSE = []
        self.Listed_sites = set()
        self.Sites_in_Path = set()
        self.Pops = set()
        self.entries = []
        self.src = filename
        self.all_sites = {}
        self.lookup_dict = None

    def populate(self, correct=True):
        '''
        Loads the file and analyses it for errors. Must be run before generating paths
        :param correct: Corrects start point and end point errors in PCM Paths if true.
        :return:Nothing
        '''

        stream = io.StringIO(self.src.stream.read().decode("UTF8", errors='ignore'), newline=None)
        reader = csv.reader(stream)
        rows = []

        for row in reader:
            rows.append(row)
        self.entries = [(r[0], r[1], r[2].split('-')) for r in rows]
        # Pop ID Mismatch
        self.PIM = [(r[1], r[2][0]) for r in self.entries if r[1] != r[2][0]]
        # Site ID Mismatch
        self.SIM = [(r[0], r[2][-1]) for r in self.entries if r[0] != r[2][-1]]
        # Replace first and last entries in PCM Path with with POP ID and Site ID
        if correct:
            self.entries = [(r[0], r[1], [r[1]] + r[2][1:-1] + [r[0]]) for r in self.entries]
        # Missing Sites
        self.Listed_sites = set()
        self.Sites_in_Path = set()
        Intermediate_points = set()
        self.Pops = set()
        for pts in self.entries:
            self.Listed_sites.add(pts[0])
            self.Pops.add(pts[1])
            for sts in pts[2]:
                self.Sites_in_Path.add(sts)
            for sts in pts[2][1:-1]:
                Intermediate_points.add(sts)
        self.Sites_in_Path |= (self.Listed_sites | self.Pops)
        diff = Intermediate_points - self.Listed_sites
        self.MSE = [r[0] for r in self.entries if (len(set(r[2][1:-1]) - diff) > 0)]
        print("There were", len(
            self.PIM), "cases where POP ID did not match with the first element of PCM Path and")
        print(len(self.SIM), "cases where Site id did not match with the last element of PCM Path.")
        if correct:
            print("All these have been rectified.")
        else:
            print("If both the numbers above are not zero, you are playing with fire.")

        try:
            print("Further, there were", len(diff), "sites that appeared as intermediate nodes in PCM path that had no")
            print("entry of their own(Ghost Sites). This would affect", len(self.MSE), "rows in all.")
        except:
            pass

    def generate_paths(self, set_Internal=False):
        '''
        generate all the paths within the network. This is important for rendering of network.
        :param set_Internal: If set to true, the script attempts to set up links between Ghost sites too.
                             Note that this is heuristic and may fail at times
        :return: Nothing
        '''
        self.all_sites = {}
        for x in self.Sites_in_Path:
            self.all_sites[x] = Site(x)
        for x in self.Pops:
            self.all_sites[x].pop = "self"
            self.all_sites[x].parent = "self"
            self.all_sites[x].verified = True
        # loop through the data one more time, just to set up the links.
        for sts in self.entries:
            self.all_sites[sts[0]].pop = sts[1]
            self.all_sites[sts[0]].parent = sts[2][-2]
            self.all_sites[sts[0]].verified = True
            self.all_sites[sts[1]].descendants.add(sts[0])
            self.all_sites[sts[2][-2]].children.add(sts[0])
            if set_Internal:
                # look at every position in the pcm path and create adjacencies. Ideally, should not be required.
                for i in range(1, len(sts[2]) - 1):
                    if self.all_sites[sts[2][i]].parent is None:
                        self.all_sites[sts[2][i]].parent = sts[2][i - 1]
                        self.all_sites[sts[2][i - 1]].children.add(sts[2][i])
                        self.all_sites[sts[2][i]].pop = sts[1]
                        self.all_sites[sts[1]].descendants.add(sts[2][i])

    def render(self, pop_id):
        '''
        Draw the graph for a certain POP
        :param pop_id: ID of pop for which graph is to be rendered
        :return:networkx digraph corresponding to selected pop
        '''
        if pop_id in self.Pops:
            DG = networkx.DiGraph()
            DG.add_node(pop_id)
            for st in self.all_sites[pop_id].descendants:
                DG.add_node(self.all_sites[st].site_id)
                DG.add_edge(self.all_sites[st].parent, self.all_sites[st].site_id)
            networkx.draw(DG, with_labels=True, node_size=1000)
            plt.show()

    def read_coordinates(self, file_name, new_format=False):
        print(new_format)
        '''
        Reads lattitude and longitude from appropriate csv file. Please remove header row before calling
        :param file_name: file object
        :return: Nothing
        '''
        if new_format:
            self.lookup_dict = {}
            stream = io.StringIO(file_name.stream.read().decode("UTF8", errors='ignore'), newline=None)
            reader = csv.reader(stream)
            for row in reader:
                if row[1] in self.all_sites:
                    try:
                        self.all_sites[row[1]].lat = float(row[3])
                        self.all_sites[row[1]].long = float(row[4])
                    except:
                        try:
                            print("Lattitude/Longitude not loaded for", row[1])
                        except:
                            pass
                    self.lookup_dict[row[0]] = (row[1], row[2] == "TDD")
            return
        coord_dict = {}
        reader = csv.reader(file_name)
        for row in reader:
            if row[3][-4:] not in coord_dict:
                coord_dict[row[3][-4:]] = (float(row[7]), float(row[8]))
        count = 0
        for s_id, s in list(self.all_sites.items()):
            if s.padded_id() in coord_dict:
                s.lat, s.long = coord_dict[s.padded_id()]
                count += 1
        print("Latitude and Longitude was added for", count, "sites.")

    def load_kpi(self, file_list, hard_clean=False, start_date=None, end_date=None):
        """
        Load KPI Data from a given set of files. Time series data for one site id Must be in a single continuous block,
        otherwise only the last block shall be retained.
        If an EU-Tran Column is available, then it is loaded else filled with 0
        If the hard clean flag is set to true, start_date and end_date need to be provided . In that case all missing
        data points shall be filled in by 1st order interpolation. This will slow down the loading process significantly
        :param file_list: list of all files to read.
        :param hard_clean: If set to true, all missing data will be filled in by first order interpolation
        :param start_date: Required for hard clean. Earliest time for which data is provided(inclusive).
        :param end_date:   Required for hard clean. Latest time for which data is provided(exclusive).
        :return:Nothing
        """
        if hard_clean:
            if start_date is None or end_date is None:
                print("Hard cleaning does not work without start and end date being set. Hard clean won't take place")
                hard_clean = False
            elif end_date < start_date:
                print("Once time-travel becomes a reality, those start and end dates will make sense. Till then,")
                print("hard clean won't be performed")
                hard_clean = False
        rows = []
        for f in file_list:
            stream = io.StringIO(f.stream.read().decode("UTF8", errors='ignore'), newline=None)
            reader = csv.reader(stream)
            for row in reader:
                rows.append(
                    [datetime.strptime(row[0], "%m.%d.%Y %H:%M:%S"), row[1][-6:],
                     float("0" + row[4].replace(',', '')),
                     float("0" + row[5].replace(',', '')),
                     float(row[6].replace(',', '')) if row[6] is not '' else -1,
                     (int(row[7]) if len(row) > 7 else 0)])
                if self.lookup_dict is not None:
                    rows[-1][1] = row[1]
        last = None
        last_id = None
        type_TDD = None
        for row in rows:
            # check if we have changed bts
            if row[1] == last and last_id is not None:
                self.all_sites[last_id].add(row, type_TDD)
            else:
                # clean up previous entry
                if last_id is not None:
                    self.all_sites[last_id].cleanup(hard_clean, start_date, end_date)
                # find if new ID exists
                if self.lookup_dict is None:
                    if row[1][2:] in self.all_sites:
                        self.all_sites[row[1][2:]].initiate(row)
                        last = row[1]
                        last_id = row[1][2:]
                    elif row[1][3:] in self.all_sites and row[1][2] == "0":
                        self.all_sites[row[1][3:]].initiate(row)
                        last = row[1]
                        last_id = row[1][3:]
                    elif row[1][4:] in self.all_sites and row[1][2:4] == "00":
                        self.all_sites[row[1][4:]].initiate(row)
                        last = row[1]
                        last_id = row[1][4:]
                    elif row[1][5:] in self.all_sites and row[1][2:5] == "000":
                        self.all_sites[row[1][5:]].initiate(row)
                        last = row[1]
                        last_id = row[1][5:]
                    else:
                        last_id = None
                        last = None
                else:
                    if row[1] in self.lookup_dict:
                        last = row[1]
                        last_id, type_TDD = self.lookup_dict[row[1]]
                        self.all_sites[last_id].initiate(row, type_TDD)
                    else:
                        last_id = None
                        last = None

        if last_id is not None:
            self.all_sites[last_id].cleanup()

    def Plot_Aggr(self, start_time, end_time, site_list, c_id, tp=0, limit=150000, Title=""):
        """
        Plots aggregate for all sites provided. Mode of aggregation is sum irrespective of nature of data
        :param start_time: datetime.datetime object or (year,month,date) tuple reflecting start time for slice(inclusive)
        :param end_time: datetime.datetime object or (year,month,date) tuple reflectimg end time for slice(exclusive)
        :param site_list: List of IDs of all sites to be aggregated
        :param ID of data to plot. 0 for S1, 1 for X2, 2 for PDCP Data Rate, 3 for Max Act UEs
        :param tp: The Type of data required, 0 for fdd,1 for tdd, 2 for combined.
        :param limit: Draws a red line at this value. Set ta a negative value to ignore
        :param Title: Anything to be added to the title. Better left blank
        :return: Pyplot.figure object containing requested plot
        """
        if type(start_time) == tuple:
            start_time = datetime(start_time[0], start_time[1], start_time[2])
        if type(end_time) == tuple:
            end_time = datetime(end_time[0], end_time[1], end_time[2])
        if start_time > end_time:
            raise Exception("Start time is after end time")
        data = []
        timeaxis = None
        delta = end_time - start_time
        expected_Entries = delta.days * 24 + delta.seconds // 3600
        fig = plt.figure(figsize=(8, 6), dpi=100)
        plt.xlabel("Time")

        # go through site_list and retrieve required data. Also perform sanity checks here
        for s in site_list:
            if s in self.all_sites:
                if tp < 2:
                    if tp == 0:
                        ret = self.all_sites[s].data_between(start_time, end_time)
                    else:
                        ret = self.all_sites[s].data_between(start_time, end_time, False)
                    if ret is not None and ret.shape[0] < expected_Entries:
                        print("Site", s, "returned only", ret.shape[
                            0], "entries instead of the expected", expected_Entries)
                        print("This site will be left out of calculations.")
                    elif ret is None:
                        print("Site", s, "has no KPI Data. It will be ignored")
                    else:
                        if timeaxis is None:
                            timeaxis = ret[:, 0]
                        data.append(ret[:, c_id + 1])
                # in case of "both" we need to add
                else:
                    ret1 = self.all_sites[s].data_between(start_time, end_time)
                    ret0 = self.all_sites[s].data_between(start_time, end_time, False)
                    if ret1 is not None and ret1.shape[0] < expected_Entries:
                        print("Site", s, "returned only", ret1.shape[
                            0], "fdd entries instead of the expected", expected_Entries)
                        print("This sites fdd component will be left out of calculations.")
                    elif ret1 is None:
                        print("Site", s, "has no fdd KPI Data. It will be ignored")
                    else:
                        if timeaxis is None:
                            timeaxis = ret1[:, 0]
                        data.append(ret1[:, c_id + 1])
                    if ret0 is not None and ret0.shape[0] < expected_Entries:
                        print("Site", s, "returned only", ret0.shape[
                            0], "tdd entries instead of the expected", expected_Entries)
                        print("This sites tdd component will be left out of calculations.")
                    elif ret0 is None:
                        print("Site", s, "has no fdd KPI Data. It will be ignored")
                    else:
                        if timeaxis is None:
                            timeaxis = ret0[:, 0]
                        data.append(ret0[:, c_id + 1])
            else:
                print("No site with ID", s, "was found in the loaded sites. This entry is being ignored.")
        if len(data) > 0:
            print("A total of", len(data), "columns were retrieved.")
            agg_data = sum(data)
            # Convert to mean for S1 and X1
            if c_id < 2:
                agg_data = agg_data / float(len(data))
            # Convert to MB for data rate
            if c_id == 2:
                agg_data = agg_data / 1000
            fig.suptitle("Aggregated Data " + Title)
            plt.plot(timeaxis, agg_data)
            plt.ylabel(
                ["S1 interface SCTP successful transmission ratio(%)",
                 "X2 interface SCTP successful transmission ratio",
                 "PDCP_DATA_RATE_MAX_DL (M8012C25)", "Maximum Active UEs with data in the buffer per cell DL"][
                    c_id])
            if c_id == 2 and limit > 0:
                plt.axhline(limit * len(data), color="red")
            #plt.show()
            return fig
        else:
            raise Exception("It appears like None of the sites returned good data")

    def Plot_POP_Aggr(self, start_time, end_time, POP_ID, c_id, limit=150000, tp=0):
        """
        Plot Aggregate data for all descendants of a given POP.
        :param start_time: datetime.datetime object or (year,month,date) tuple reflecting start time for slice(inclusive)
        :param end_time: datetime.datetime object or (year,month,date) tuple reflectimg end time for slice(exclusive)
        :param POP_ID: ID of the pop for which data is to be aggregated
        :param c_id: ID of data to plot. 0 for S1, 1 for X2, 2 for PDCP Data Rate
        :param tp: The Type of data required, 0 for fdd,1 for tdd, 2 for combined.
        :return: Pyplot.figure object containing requested plot
        """
        if type(start_time) == tuple:
            start_time = datetime(start_time[0], start_time[1], start_time[2])
        if type(end_time) == tuple:
            end_time = datetime(end_time[0], end_time[1], end_time[2])

        if POP_ID in self.Pops:
            return self.Plot_Aggr(start_time, end_time, list(self.all_sites[POP_ID].descendants) + [POP_ID], c_id,
                                  tp, limit,
                                  "For POP " + POP_ID)
        else:
            raise Exception("This POP id is not available")

    def Load_Link_Data(self, link_file, update_lat_long=True, new_format=True):
        """
        The main purpose of this function is to Load bandwidth data for various sites. For this to work correctly, the
        csv file should have atleast the following data in following column(indexes begin at 0):
        Column No   Excel Column Name   Data Required
        4           E                   Bandwidth
        7           H                   Site ID of First Site
        11          L                   Latitude of First Site*
        12          M                   Longitude of First Site*
        26          AA                  Site ID of Second Site
        30          AE                  Latitude of Second Site*
        31          AF                  Longitude of Second Site*
        If the Star marked fields are present and the corresponding flag is set to true, this will also load any
        new lattitude-longitude data that it encounters. If this data is however incoherent with previous data, previous
        data will be retained.

        Further, it is assumed that Bandwidth will be in the form "X MB" where X is a valid regex integer.
        :param link_file:Path to csv File pertaining to the above format
        :param update_lat_long: Set to false if star marked fields are unavailable. In that case, Latitude and Longitude
        will not be updated
        :return:Nothing
        """

        def isfloat(value):
            try:
                float(value)
                return True
            except ValueError:
                return False

        stream = io.StringIO(link_file.stream.read().decode("UTF8", errors='ignore'), newline=None)
        reader = csv.reader(stream)
        data = []
        if update_lat_long:
            for row in reader:
                if new_format:
                    data.append([row[1], row[2], row[5], row[3], row[4], row[6], row[7]])
                else:
                    data.append([row[4], row[7], row[26], row[11], row[12], row[30], row[31]])
        else:
            for row in reader:
                if new_format:
                    data.append([row[1], row[2], row[5]])
                else:
                    data.append([row[4], row[7], row[26]])
        ban_added = 0
        ll_updated = 0
        for entry in data:
            # link could be either way. Check for both possibilities. Insert into childs uplink if found
            if entry[1] in self.all_sites and self.all_sites[entry[1]].parent == entry[2]:
                self.all_sites[entry[1]].uplink = int(entry[0][:-3])
                ban_added += 1
            elif entry[2] in self.all_sites and self.all_sites[entry[2]].parent == entry[1]:
                self.all_sites[entry[2]].uplink = int(entry[0][:-3])
                ban_added += 1
            if update_lat_long:
                if entry[1] in self.all_sites and self.all_sites[entry[1]].lat == 0 and self.all_sites[
                    entry[1]].long == 0 and isfloat(entry[3]) and isfloat(entry[4]):
                    self.all_sites[entry[1]].lat = float(entry[3])
                    self.all_sites[entry[1]].long = float(entry[4])
                    ll_updated += 1
                if entry[2] in self.all_sites and self.all_sites[entry[2]].lat == 0 and self.all_sites[
                    entry[2]].long == 0 and isfloat(entry[5]) and isfloat(entry[6]):
                    self.all_sites[entry[2]].lat = float(entry[5])
                    self.all_sites[entry[2]].long = float(entry[6])
                    ll_updated += 1
        print("Bandwidth data loaded for", ban_added, "sites")
        if update_lat_long:
            print("Coordinates loaded for a further", ll_updated, "sites")

    def propagate(self, tp=2):
        """
        Propagate KPIs from child to POP to find out load on each link. For this to work correctly, data should be
        either noise-free of hard cleaned.
        :param tp: The Type of data required, 0 for fdd,1 for tdd, 2 for combined.
        :return:
        """
        for K in self.Pops:
                self.all_sites[K].propagate(self.all_sites, tp)

    def set_occupancy(self):
        """
        Set the max_Occupancy parameter where it can be set. Must go after loading link data and propagating
        :return: Nothing
        """
        for K in self.all_sites:
            if self.all_sites[K].uplink > 0 and self.all_sites[K].kpi_propagated:
                self.all_sites[K].max_occupancy = max(self.all_sites[K].downstream_kpi[:, 1]) / self.all_sites[
                    K].uplink / 1000

    def get_faults(self, limit=150000, tp=2, POP=False):
        """
        Check which sites are exceeding the given limit
        :param limit: The limit...Simple, right
        :param: POP data
        :param tp: The Type of data required, 0 for fdd,1 for tdd, 2 for combined.
        :return: List of sites exceeding the limit.
        """
        res = []
        for x in self.all_sites:
            if self.all_sites[x].check_limit(limit, tp, POP):
                res.append([x, self.all_sites[x].pop])
        return res


def pretty_load(path_file, KPI_files, Data_file, coord_file="", hard_clean=False, start_date=None, end_date=None):
    """
    Short-hand for the entire input parsing process. On error, it might be a good idea to try the step-by-step way
    :param: path_file: Path to csv file containing the network architecture
    :param:  KPI_files: Paths to csv files containing KPI Data
    :param: coord_file: Path to csv files containing coordinate data. Ignored if left blank.
    :param: Data_file: Path to csv file containing bandwidth data
    :return: Created Site.Network object
    """
    if type(start_date) == tuple:
        start_date = datetime(start_date[0], start_date[1], start_date[2])
    if type(end_date) == tuple:
        end_date = datetime(end_date[0], end_date[1], end_date[2])
    X = Network(path_file)
    X.populate()
    X.generate_paths(True)
    if not coord_file == "":
        X.read_coordinates(coord_file)
    print("Now Loading KPI.")
    X.load_kpi(KPI_files, hard_clean, start_date, end_date)
    X.Load_Link_Data(Data_file)
    X.propagate()
    X.set_occupancy()
    return X

