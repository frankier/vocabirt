import os
from os.path import join as pjoin

import click
import duckdb
import pycountry

QUERY = """
PRAGMA memory_limit='4GB';
PRAGMA set_progress_bar_time=1;

SELECT answers.user_id AS user_id, ranks.word AS word, answers.known AS known
FROM users
JOIN answers
ON users.user_id = answers.user_id
JOIN ranks
ON answers.rank = ranks.rank
WHERE {}
ORDER BY answers.user_id;
"""

country_lookup_pairs = [
    line.split(" ", 1)
    for line in """
1 Afghanistan
2 Albania
3 Algeria
4 Andorra
5 Angola
6 Antigua & Deps
7 Argentina
8 Armenia
9 Australia
10 Austria
11 Azerbaijan
12 Bahamas
13 Bahrain
14 Bangladesh
15 Barbados
16 Belarus
17 Belgium
18 Belize
19 Benin
20 Bhutan
21 Bolivia
22 Bosnia Herzegovina
23 Botswana
24 Brazil
25 Brunei
26 Bulgaria
27 Burkina
28 Burundi
29 Cambodia
30 Cameroon
31 Canada
32 Cape Verde
33 Central African Rep
34 Chad
35 Chile
36 China
37 Colombia
38 Comoros
39 Congo
40 Congo (Democratic Rep)
41 Costa Rica
42 Croatia
43 Cuba
44 Cyprus
45 Czech Republic
46 Denmark
47 Djibouti
48 Dominica
49 Dominican Republic
50 East Timor
51 Ecuador
52 Egypt
53 El Salvador
54 Equatorial Guinea
55 Eritrea
56 Estonia
57 Ethiopia
58 Fiji
59 Finland
60 France
61 Gabon
62 Gambia
63 Georgia
64 Germany
65 Ghana
66 Greece
67 Grenada
68 Guatemala
69 Guinea
70 Guinea-Bissau
71 Guyana
72 Haiti
73 Honduras
74 Hungary
75 Iceland
76 India
77 Indonesia
78 Iran
79 Iraq
80 Ireland (Republic)
81 Israel
82 Italy
83 Ivory Coast
84 Jamaica
85 Japan
86 Jordan
87 Kazakhstan
88 Kenya
89 Kiribati
90 Korea, North
91 Korea, South
92 Kosovo
93 Kuwait
94 Kyrgyzstan
95 Laos
96 Latvia
97 Lebanon
98 Lesotho
99 Liberia
100 Libya
101 Liechtenstein
102 Lithuania
103 Luxembourg
104 Macedonia
105 Madagascar
106 Malawi
107 Malaysia
108 Maldives
109 Mali
110 Malta
111 Marshall Islands
112 Mauritania
113 Mauritius
114 Mexico
115 Micronesia
116 Moldova
117 Monaco
118 Mongolia
119 Montenegro
120 Morocco
121 Mozambique
122 Myanmar (Burma)
123 Namibia
124 Nauru
125 Nepal
126 Netherlands
127 New Zealand
128 Nicaragua
129 Niger
130 Nigeria
131 Norway
132 Oman
133 Pakistan
134 Palau
135 Panama
136 Papua New Guinea
137 Paraguay
138 Peru
139 Philippines
140 Poland
141 Portugal
142 Qatar
143 Romania
144 Russian Federation
145 Rwanda
146 St Kitts & Nevis
147 St Lucia
148 Saint Vincent & the Grenadines</option>
149 Samoa
150 San Marino
151 Sao Tome & Principe
152 Saudi Arabia
153 Senegal
154 Serbia
155 Seychelles
156 Sierra Leone
157 Singapore
158 Slovakia
159 Slovenia
160 Solomon Islands
161 Somalia
162 South Africa
163 Spain
164 Sri Lanka
165 Sudan
166 Sudan, South
167 Suriname
168 Swaziland
169 Sweden
170 Switzerland
171 Syria
172 Taiwan
173 Tajikistan
174 Tanzania
175 Thailand
176 Togo
177 Tonga
178 Trinidad & Tobago
179 Tunisia
180 Turkey
181 Turkmenistan
182 Tuvalu
183 Uganda
184 Ukraine
185 United Arab Emirates
186 United Kingdom
187 United States
188 Uruguay
189 Uzbekistan
190 Vanuatu
191 Vatican City
192 Venezuela
193 Vietnam
194 Yemen
195 Zambia
196 Zimbabwe
""".strip().split(
        "\n"
    )
]

country_lookup = {}
for country_id, country_name in country_lookup_pairs:
    try:
        countries = pycountry.countries.search_fuzzy(country_name)
    except LookupError:
        continue
    if not len(countries):
        continue
    country_lookup[countries[0].alpha_2.lower()] = int(country_id)


def save_query(db, query, path):
    print("Running")
    print(query)
    df = db.execute(query).fetchdf()
    print()
    print("Saving to", path)
    df.to_parquet(path)


@click.command()
@click.argument("dbfn", type=click.Path())
@click.argument("out", type=click.Path())
@click.option("--split")
def main(dbfn, out, split):
    db = duckdb.connect(dbfn)
    if split == "val":
        query = QUERY.format(
            "year(users.datetime) = 2019 AND month(users.datetime) <= 9"
        )
        save_query(db, query, out)
    elif split.startswith("test"):
        if len(split) > 4:
            nationality = country_lookup[split[4:]]
        else:
            nationality = 59
        query = QUERY.format(
            f"users.nationality_nonnative = {nationality} AND year(users.datetime) = 2018"
        )
        save_query(db, query, out)
    else:
        os.makedirs(out, exist_ok=True)
        for year in range(2011, 2018):
            for month in range(1, 13):
                path = pjoin(out, f"{year}_{month:02d}.arrow")
                query = QUERY.format(
                    f"year(users.datetime) = {year} AND month(users.datetime) = {month}"
                )
                save_query(db, query, path)


if __name__ == "__main__":
    main()
