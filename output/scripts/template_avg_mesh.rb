###------------------ GEMS ------------------###
require 'gnuplot'

###---------- SCRIPT CONFIGURATION ----------###

# data file parameters
FIELD_DIR = "../field"
POTENTIAL_DIR = "../potential"
AVERAGED = true
BOT = 3000000
TOP = 5000000
STEP = 1000
NODES = 1000
GAMMA = 1000
H = 0.1
V0 = -0.003989423 #-0.014142136
PI = Math::PI

# plot parameters 
param_E2_vs_Phi = {:title => "Squared field vs potential (data averaged over #{TOP-BOT} iterations, from iteration #{BOT} to #{TOP})", 
                   :xlabel => "Potential (measured in simulation units)",
                   :ylabel => "Squared electric field (measured in simulation units)"}
param_Phi = {:title => "Stationary potential distribution (data averaged over #{TOP-BOT} iterations, from iteration #{BOT} to #{TOP})", 
             :xlabel => "Distance from probe (measured in Debye lenghts units)",
             :ylabel => "Potential (measured in simulation units)"}
param_E = {:title => "Stationary field distribution (data averaged over #{TOP-BOT} iterations, from iteration #{BOT} to #{TOP})", 
           :xlabel => "Distance from probe (measured in Debye lenghts units)",
           :ylabel => "Electric field (measured in simulation units)"}

###-------------- SCRIPT START --------------###

# format of the file names (KEY will be changed by the value of iter or counter respectively)
if AVERAGED
  IFNAME_E = "#{FIELD_DIR}/avg_field_t_KEY.dat"
  IFNAME_PHI = "#{POTENTIAL_DIR}/avg_potential_t_KEY.dat"
  OFNAME_E2_VS_PHI = "squared_field_vs_potential_from_#{BOT}_to_#{TOP}.tex"
  OFNAME_PHI = "potential_vs_position_from_#{BOT}_to_#{TOP}.tex"
  OFNAME_E = "field_vs_position_from_#{BOT}_to_#{TOP}.tex"
else
  IFNAME_E = "#{FIELD_DIR}/field_t_KEY.dat"
  IFNAME_PHI = "#{POTENTIAL_DIR}/potential_t_KEY.dat"
  OFNAME_E2_VS_PHI = "squared_field_vs_potential_from_#{BOT}_to_#{TOP}.tex"
  OFNAME_PHI = "potential_vs_potential_from_#{BOT}_to_#{TOP}.tex"
  OFNAME_E = "field_vs_potential_from_#{BOT}_to_#{TOP}.tex"
end

#------------------------------------------------------

#---- Read field and potential data from simulation data files
field_data = Array.new(NODES+1, 0.0)
field2_data = Array.new(NODES+1, 0.0)
potential_data = Array.new(NODES+1, 0.0)
position = Array.new(NODES+1, 0.0)

(BOT..TOP).step(STEP) do |iter|
  f1 = File.open("#{IFNAME_E.gsub("KEY", iter.to_s)}", mode="r")
  f2 = File.open("#{IFNAME_PHI.gsub("KEY", iter.to_s)}", mode="r")
  a = f1.readlines
  b = f2.readlines
  (0..NODES).step do |index|
    position[index] = a[index].split[0].to_f/10.0
    field_data[index] += a[index].split[1].to_f
    potential_data[index] += b[index].split[1].to_f
  end
  f1.close
  f2.close
end

#---- Normalize data and store file with graphs
f1 = File.open("averaged_mesh_data.dat", mode="w")
(0..NODES).step do |index|
  field_data[index] = field_data[index]/((TOP-BOT)/STEP+1).to_f
  field2_data[index] = field_data[index]**2
  potential_data[index] = potential_data[index]/((TOP-BOT)/STEP+1).to_f
  f1.write("#{index} #{position[index]} #{potential_data[index]} #{field_data[index]} #{field2_data[index]}\n")
end
f1.close

#---- Obtain probe data
PHI_P = potential_data[0]
E_P = field_data[0]
E_P2 = field2_data[0]

#---- Solve fluid model with RK4
field_model = Array.new(NODES+1, 0.0)
field2_model = Array.new(NODES+1, 0.0)
potential_model = Array.new(NODES+1, 0.0)
position_model = Array.new(NODES+1, 0.0)

modifier = 1.0e-4
potential_model[0] = PHI_P
field_model[0] = E_P
field_model[NODES] = field_data[NODES]
potential_model[NODES-1] = -10.0
position_model[0] = 0

def fPhi(e) 
  return -e
end

def fE(phi)
  return 1.0/(Math.sqrt(1.0-2.0*phi/(GAMMA*V0*V0)))-0.5*Math.exp(phi)*(1.0+Math.erf(Math.sqrt(phi-PHI_P)))
end

(1..10000).step do |iter|
  (1..NODES-1).step do |index|
    position_model[index] = index*H

    k1Phi = fPhi(field_model[index-1])
    k1E = fE(potential_model[index-1])

    k2Phi = fPhi(field_model[index-1]+H*0.5*k1E)
    k2E = fE(potential_model[index-1]+H*0.5*k1Phi)

    k3Phi = fPhi( field_model[index-1]+H*0.5*k2E)
    k3E = fE(potential_model[index-1]+H*0.5*k2Phi)

    k4Phi = fPhi(field_model[index-1]+H*k3E)
    k4E = fE(potential_model[index-1]+H*k3Phi)

    potential_model[index] = potential_model[index-1]+H*(k1Phi+2.0*k2Phi+2.0*k3Phi+k4Phi)/6.0
    field_model[index] = field_model[index-1]+H*(k1E+2.0*k2E+2.0*k3E+k4E)/6.0

    if (index < NODES-1 && potential_model[index] > -0.15) 
      field_model[0] += modifier
      puts index
      break
    elsif (index == NODES-1 && potential_model[index] < -0.1)
      field_model[0] -= modifier
      puts index
    elsif (index > 2 && potential_model[index] < potential_model[index-2]) 
      field_model[0] -= modifier
      puts index
      break
    end
  end

  if (iter % 100 == 0)
    modifier *= 0.1
  end
  if (potential_model[NODES-1] > -0.1)
    break
  end
end

(0..NODES).step do |index|
  field2_model[index] = field_model[index]**2
end

#---- Plot results
Gnuplot.open do |gp|
 
  # E2 vs Phi
  Gnuplot::Plot.new(gp) do |plot|
    plot.terminal "epslatex size 8,6 standalone color colortext 10"
    #plot.nokey
    plot.grid
    plot.ylabel param_E2_vs_Phi[:ylabel]
    plot.xlabel param_E2_vs_Phi[:xlabel]
    plot.title param_E2_vs_Phi[:title]
    plot.output OFNAME_E2_VS_PHI
    plot.data = [
      Gnuplot::DataSet.new( [potential_data, field2_data] ) { |ds|
        ds.with = "points ps 2"
        ds.title = "PIC Simulations"
        ds.linewidth = 4
      }, 
      Gnuplot::DataSet.new( "#{E_P2}+exp(x)*(erf(sqrt(x-#{PHI_P}))+1.)-exp(#{PHI_P})*(1.+2.*sqrt(x-#{PHI_P})/sqrt(#{PI}))+2.*#{V0}*#{V0}*#{GAMMA}*(sqrt(1.-2.*x/(#{GAMMA}*#{V0}*#{V0}))-sqrt(1.-2.*#{PHI_P}/(#{GAMMA}*#{V0}*#{V0})))" ) { |ds|
        ds.with = "lines lt 1"
        ds.title = "Fluid model (analitic)"
        ds.linecolor = 'rgb "blue"'
        ds.linewidth = 4
      },
      Gnuplot::DataSet.new( [potential_model, field2_model] ) { |ds|
        ds.with = "lines lt 2"
        ds.title = "Fluid model (numeric)"
        ds.linecolor = 'rgb "green"'
        ds.linewidth = 4
      }
    ]
  end

  # Potential
  Gnuplot::Plot.new(gp) do |plot|
    plot.terminal "epslatex size 8,6 standalone color colortext 10"
    #plot.nokey
    plot.grid
    plot.ylabel param_Phi[:ylabel]
    plot.xlabel param_Phi[:xlabel]
    plot.title param_Phi[:title]
    plot.output OFNAME_PHI
    plot.data = [
      Gnuplot::DataSet.new( [position, potential_data] ) { |ds|
        ds.with = "points ps 2"
        ds.title = "PIC Simulations"
        ds.linewidth = 4
      }, 
      Gnuplot::DataSet.new( [position, potential_model] ) { |ds|
        ds.with = "lines lt 1"
        ds.title = "Fluid model (numeric)"
        ds.linecolor = 'rgb "blue"'
        ds.linewidth = 4
      }
    ]
  end

  # Field 
  Gnuplot::Plot.new(gp) do |plot|
    plot.terminal "epslatex size 8,6 standalone color colortext 10"
    #plot.nokey
    plot.grid
    plot.ylabel param_E[:ylabel]
    plot.xlabel param_E[:xlabel]
    plot.title param_E[:title]
    plot.output OFNAME_E
    plot.data = [
      Gnuplot::DataSet.new( [position, field_data] ) { |ds|
        ds.with = "points ps 2"
        ds.title = "PIC Simulations"
        ds.linewidth = 4
      }, 
      Gnuplot::DataSet.new( [position, field_model] ) { |ds|
        ds.with = "lines lt 1"
        ds.title = "Fluid model (numeric)"
        ds.linecolor = 'rgb "blue"'
        ds.linewidth = 4
      }
    ]
  end

end

`pdflatex #{OFNAME_E2_VS_PHI}`
`pdflatex #{OFNAME_E}`
`pdflatex #{OFNAME_PHI}`
`rm *.aux *.log *.tex *.eps *converted-to.pdf`
