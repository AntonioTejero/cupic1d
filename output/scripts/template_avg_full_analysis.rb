###------------------ GEMS ------------------###
require 'gnuplot'

###---------- SCRIPT CONFIGURATION ----------###

# data file parameters
FIELD_DIR = "../field"
POTENTIAL_DIR = "../potential"
PARTICLE_DIR = "../particles"
LOG_DIR = ".."
BOT = 3000000
TOP = 5000000
STEP = 1000
NODES = 1000
BINS = 100
GAMMA = 1000
N = 328637.9
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
param_ddf = {:title => "Density distribution functions (data averaged over #{TOP-BOT} iterations, from iteration #{BOT} to #{TOP})", 
             :xlabel => "Position (measured in simulation units)",
             :ylabel => "Counts"}
param_meanv_e = {:title => "Mean velocity of electrons (data averaged over #{TOP-BOT} iterations, from iteration #{BOT} to #{TOP})", 
                 :xlabel => "Position (measured in simulation units)",
                 :ylabel => "Velocity (measured in simulation units)"}
param_meanv_i = {:title => "Mean velocity of ions (data averaged over #{TOP-BOT} iterations, from iteration #{BOT} to #{TOP})", 
                 :xlabel => "Position (measured in simulation units)",
                 :ylabel => "Velocity (measured in simulation units)"}
param_vdf_e = {:title => "Electron velocity distribution functions (data averaged over #{TOP-BOT} iterations, from iteration #{BOT} to #{TOP})", 
               :xlabel => "Velocity (measured in simulation units)",
               :ylabel => "Counts"}
param_vdf_i = {:title => "Ion velocity distribution functions (data averaged over #{TOP-BOT} iterations, from iteration #{BOT} to #{TOP})", 
               :xlabel => "Velocity (measured in simulation units)",
               :ylabel => "Counts"}
param_flux_e = {:title => "Electron flux distribution (data averaged over #{TOP-BOT} iterations, from iteration #{BOT} to #{TOP})", 
                :xlabel => "Position (measured in simulation units)",
                :ylabel => "Flux (measured in simulation units)"}
param_flux_i = {:title => "Ion flux distribution (data averaged over #{TOP-BOT} iterations, from iteration #{BOT} to #{TOP})", 
                :xlabel => "Position (measured in simulation units)",
                :ylabel => "Flux (measured in simulation units)"}

###-------------- SCRIPT START --------------###

# format of the file names (KEY will be changed by the value of iter or counter respectively)
IFNAME_E = "#{FIELD_DIR}/avg_field_t_KEY.dat"
IFNAME_PHI = "#{POTENTIAL_DIR}/avg_potential_t_KEY.dat"
IFNAME_VDF_E = "#{PARTICLE_DIR}/electrons_vdf_t_KEY.dat"
IFNAME_DDF_E = "#{PARTICLE_DIR}/electrons_ddf_t_KEY.dat"
IFNAME_VDF_I = "#{PARTICLE_DIR}/ions_vdf_t_KEY.dat"
IFNAME_DDF_I = "#{PARTICLE_DIR}/ions_ddf_t_KEY.dat"
IFNAME_LOG = "#{LOG_DIR}/log.dat"

OFNAME_AVG_MESH_DATA = "./mesh_data_from_#{BOT}_to_#{TOP}.dat"
OFNAME_VDF = "./vdf_from_#{BOT}_to_#{TOP}.dat" 
OFNAME_DDF = "./ddf_from_#{BOT}_to_#{TOP}.dat"
OFNAME_FLUX = "./flux_from_#{BOT}_to_#{TOP}.dat"
OFNAME_FLUID_MODEL_DATA = "./fluid_model_data.dat"

OFNAME_E2_VS_PHI = "squared_field_vs_potential_from_#{BOT}_to_#{TOP}.tex"
OFNAME_PHI = "potential_vs_position_from_#{BOT}_to_#{TOP}.tex"
OFNAME_E = "field_vs_position_from_#{BOT}_to_#{TOP}.tex"
OFNAME_VDF_GRAPH = "./vdf_from_#{BOT}_to_#{TOP}.tex"
OFNAME_DDF_GRAPH = "./ddf_from_#{BOT}_to_#{TOP}.tex"
OFNAME_MEAN_VEL_E_GRAPH = "./mean_vel_e_from_#{BOT}_to_#{TOP}.tex"
OFNAME_MEAN_VEL_I_GRAPH = "./mean_vel_i_from_#{BOT}_to_#{TOP}.tex"
OFNAME_FLUX_I_GRAPH = "./flux_i_from_#{BOT}_to_#{TOP}.tex"
OFNAME_FLUX_E_GRAPH = "./flux_e_from_#{BOT}_to_#{TOP}.tex"

#------------------------------------------------------

#---- Read, average and store mesh data
field_data = Array.new(NODES+1, 0.0)
field2_data = Array.new(NODES+1, 0.0)
potential_data = Array.new(NODES+1, 0.0)
position_mesh = Array.new(NODES+1, 0.0)

puts"Reading mesh data from simulation: \n"
(BOT..TOP).step(STEP) do |iter|
  puts"\t Reading mesh file #{iter}...\n"
  f1 = File.open("#{IFNAME_E.gsub("KEY", iter.to_s)}", mode="r")
  f2 = File.open("#{IFNAME_PHI.gsub("KEY", iter.to_s)}", mode="r")
  a = f1.readlines
  b = f2.readlines
  (0..NODES).step do |index|
    position_mesh[index] = a[index].split[0].to_f/10.0
    field_data[index] += a[index].split[1].to_f
    potential_data[index] += b[index].split[1].to_f
  end
  f1.close
  f2.close
end

puts"Saving averaged mesh data:\n"
f1 = File.open(OFNAME_AVG_MESH_DATA, mode="w")
(0..NODES).step do |index|
  field_data[index] = field_data[index]/((TOP-BOT)/STEP+1).to_f
  field2_data[index] = field_data[index]**2
  potential_data[index] = potential_data[index]/((TOP-BOT)/STEP+1).to_f
  f1.write("#{index} #{position_mesh[index]} #{potential_data[index]} #{field_data[index]} #{field2_data[index]}\n")
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

modifier = 1.0e-4
potential_model[0] = PHI_P
field_model[0] = E_P
field_model[NODES] = field_data[NODES]
potential_model[NODES-1] = -10.0

def fPhi(e) 
  return -e
end

def fE(phi)
  return 1.0/(Math.sqrt(1.0-2.0*phi/(GAMMA*V0*V0)))-0.5*Math.exp(phi)*(1.0+Math.erf(Math.sqrt(phi-PHI_P)))
end

puts"Solving fluid model with shooting method:\n"
(1..10000).step do |iter|
  (1..NODES-1).step do |index|

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
      puts"\t Reached node -> #{index}\n"
      break
    elsif (index == NODES-1 && potential_model[index] < -0.1)
      field_model[0] -= modifier
      puts"\t Reached node -> #{index}\n"
    elsif (index > 2 && potential_model[index] < potential_model[index-2]) 
      field_model[0] -= modifier
      puts"\t Reached node -> #{index}\n"
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

#---- Evaluate particle densities and mean ion velocity
density_e = Array.new(NODES+1, 0.0)
density_i = Array.new(NODES+1, 0.0)
velocity_model_i = Array.new(NODES+1, 0.0)

puts"Saving fluid model data:\n"
f1 = File.open(OFNAME_FLUID_MODEL_DATA, mode="w")
(0..NODES).step do |index|
  density_e[index] = N*0.5*Math.exp(potential_model[index])*(1.0+Math.erf(Math.sqrt(potential_model[index]-PHI_P)))
  density_i[index] = N*1.0/Math.sqrt(1.0-2.0*potential_model[index]/(GAMMA*V0*V0))
  velocity_model_i[index] = -Math.sqrt(V0*V0-2.0*potential_model[index]/GAMMA)
  f1.write("#{position_mesh[index]} #{potential_model[index]} #{field_model[index]} #{field2_model[index]} #{density_e[index]} #{density_i[index]} #{velocity_model_i[index]} \n")
end
f1.close

#---- Read, average and store df data
vdf_e_data = Array.new(BINS){Array.new(BINS, 0.0)}
ddf_e_data = Array.new(BINS, 0.0)
velocity_e = Array.new(BINS, 0.0)
vdf_i_data = Array.new(BINS){Array.new(BINS, 0.0)}
ddf_i_data = Array.new(BINS, 0.0)
velocity_i = Array.new(BINS, 0.0)
position_bin = Array.new(BINS, 0.0)

puts"Reading df data from simulation: \n"
(BOT..TOP).step(STEP) do |iter|
  puts"\t Reading df file #{iter}...\n"
  f1 = File.open("#{IFNAME_VDF_E.gsub("KEY", iter.to_s)}", mode="r")
  f2 = File.open("#{IFNAME_VDF_I.gsub("KEY", iter.to_s)}", mode="r")
  f3 = File.open("#{IFNAME_DDF_E.gsub("KEY", iter.to_s)}", mode="r")
  f4 = File.open("#{IFNAME_DDF_I.gsub("KEY", iter.to_s)}", mode="r")
  a = f1.readlines
  b = f2.readlines
  c = f3.readlines
  d = f4.readlines
  (0..BINS-1).step do |binp|
    position_bin[binp] = c[binp].split[0].to_f
    ddf_e_data[binp] += c[binp].split[1].to_f
    ddf_i_data[binp] += d[binp].split[1].to_f
    (0..BINS-1).step do |binv|
      vdf_e_data[binp][binv] += a[binp*(BINS+1)+binv].split[2].to_f
      velocity_e[binv] = a[binp*(BINS+1)+binv].split[1].to_f
      vdf_i_data[binp][binv] += b[binp*(BINS+1)+binv].split[2].to_f
      velocity_i[binv] = b[binp*(BINS+1)+binv].split[1].to_f
    end
  end
  f1.close
  f2.close
  f3.close
  f4.close
end

ne = 0.0
ni = 0.0
f1 = File.open(IFNAME_LOG, mode="r")
a = f1.readlines
(BOT..TOP).step do |iter|
  ne += a[iter/STEP].split[1].to_f
  ni += a[iter/STEP].split[2].to_f
end
f1.close
ne /= (TOP-BOT+1).to_f
ni /= (TOP-BOT+1).to_f

puts"Saving averaged distribution functions:\n"
f1 = File.open(OFNAME_DDF, mode="w")
f2 = File.open(OFNAME_VDF, mode="w")
(0..BINS-1).step do |binp|
  ddf_e_data[binp] *= ne/(H*H*(TOP-BOT+STEP).to_f)
  ddf_i_data[binp] *= ni/(H*H*(TOP-BOT+STEP).to_f)
  f1.write("#{position_bin[binp]} #{ddf_e_data[binp]} #{ddf_i_data[binp]} \n")
  (0..BINS-1).step do |binv|
    vdf_e_data[binp][binv] /= (TOP-BOT+STEP).to_f
    vdf_i_data[binp][binv] /= (TOP-BOT+STEP).to_f
    f2.write("#{position_bin[binp]} #{velocity_e[binv]} #{vdf_e_data[binp][binv]} #{velocity_i[binv]} #{vdf_i_data[binp][binv]} \n")
  end
  f2.write("\n")
end
f1.close
f2.close

#---- Evaluate particle flux
mean_velocity_e = Array.new(BINS, 0.0)
mean_velocity_i = Array.new(BINS, 0.0)
flux_e = Array.new(BINS, 0.0)
flux_i = Array.new(BINS, 0.0)

puts"Evaluating particle flux:\n"
f1 = File.open(OFNAME_FLUX, mode="w")
(0..BINS-1).step do |binp|
  norm_e = 0.0
  norm_i = 0.0
  (0..BINS-1).step do |binv|
    mean_velocity_e[binp] += velocity_e[binv]*vdf_e_data[binp][binv] 
    norm_e += vdf_e_data[binp][binv] 
    mean_velocity_i[binp] += velocity_i[binv]*vdf_i_data[binp][binv] 
    norm_i += vdf_i_data[binp][binv] 
  end
  mean_velocity_e[binp] /= norm_e
  mean_velocity_i[binp] /= norm_i
  flux_e[binp] = mean_velocity_e[binp]*ddf_e_data[binp]
  flux_i[binp] = mean_velocity_i[binp]*ddf_i_data[binp]
  f1.write("#{position_bin[binp]} #{mean_velocity_e[binp]} #{flux_e[binp]} #{mean_velocity_i[binp]} #{flux_i[binp]} \n")
end
f1.close


#---- Plot results
Gnuplot.open do |gp|
 
  # E2 vs Phi
  puts "\t Ploting squared field vs potential...\n"
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
  puts "\t Ploting potential distribution...\n"
  Gnuplot::Plot.new(gp) do |plot|
    plot.terminal "epslatex size 8,6 standalone color colortext 10"
    #plot.nokey
    plot.grid
    plot.ylabel param_Phi[:ylabel]
    plot.xlabel param_Phi[:xlabel]
    plot.title param_Phi[:title]
    plot.output OFNAME_PHI
    plot.data = [
      Gnuplot::DataSet.new( [position_mesh, potential_data] ) { |ds|
        ds.with = "points ps 2"
        ds.title = "PIC Simulations"
        ds.linewidth = 4
      }, 
      Gnuplot::DataSet.new( [position_mesh, potential_model] ) { |ds|
        ds.with = "lines lt 1"
        ds.title = "Fluid model (numeric)"
        ds.linecolor = 'rgb "blue"'
        ds.linewidth = 4
      }
    ]
  end

  # Field 
  puts "\t Ploting field distribution...\n"
  Gnuplot::Plot.new(gp) do |plot|
    plot.terminal "epslatex size 8,6 standalone color colortext 10"
    #plot.nokey
    plot.grid
    plot.ylabel param_E[:ylabel]
    plot.xlabel param_E[:xlabel]
    plot.title param_E[:title]
    plot.output OFNAME_E
    plot.data = [
      Gnuplot::DataSet.new( [position_mesh, field_data] ) { |ds|
        ds.with = "points ps 2"
        ds.title = "PIC Simulations"
        ds.linewidth = 4
      }, 
      Gnuplot::DataSet.new( [position_mesh, field_model] ) { |ds|
        ds.with = "lines lt 1"
        ds.title = "Fluid model (numeric)"
        ds.linecolor = 'rgb "blue"'
        ds.linewidth = 4
      }
    ]
  end

  # Density distribution plot
  puts "\t Ploting density distributions...\n"
  Gnuplot::Plot.new(gp) do |plot|
    plot.terminal "epslatex size 8,6 standalone color colortext 10"
    #plot.nokey
    plot.grid
    plot.title param_ddf[:title]
    plot.xlabel param_ddf[:xlabel]
    plot.ylabel param_ddf[:ylabel]
    plot.output OFNAME_DDF_GRAPH
    plot.data = [
      Gnuplot::DataSet.new( [position_bin, ddf_e_data] ) { |ds|
        ds.with = "boxes lt 1"
        ds.title = "Electrons (Simulation)"
        ds.linewidth = 4
        ds.linecolor = 'rgb "red"'
      }, 
      Gnuplot::DataSet.new( [position_bin, ddf_i_data] ) { |ds|
        ds.with = "boxes lt 1"
        ds.title = "Ions (Simulation)"
        ds.linewidth = 4
        ds.linecolor = 'rgb "blue"'
      },
      Gnuplot::DataSet.new( [position_mesh, density_e] ) { |ds|
        ds.with = "lines lt 1"
        ds.title = "Electron (fluid model prediction)"
        ds.linecolor = 'rgb "red"'
        ds.linewidth = 4
      }, 
      Gnuplot::DataSet.new( [position_mesh, density_i] ) { |ds|
        ds.with = "lines lt 1"
        ds.title = "Ion (fluid model prediction)"
        ds.linecolor = 'rgb "blue"'
        ds.linewidth = 4
      }
    ]
  end

  # Mean velocitys of electrons
  puts "\t Ploting mean velocity of electrons...\n"
  Gnuplot::Plot.new(gp) do |plot|
    plot.terminal "epslatex size 8,6 standalone color colortext 10"
    #plot.nokey
    plot.grid
    plot.title param_meanv_e[:title]
    plot.xlabel param_meanv_e[:xlabel]
    plot.ylabel param_meanv_e[:ylabel]
    plot.output OFNAME_MEAN_VEL_E_GRAPH 
    plot.data = [
      Gnuplot::DataSet.new( [position_bin, mean_velocity_e] ) { |ds|
        ds.with = "lines lt 1"
        ds.title = "Electrons"
        ds.linewidth = 4
        ds.linecolor = 'rgb "red"'
      }
    ]
  end

  # Mean velocitys of ions
  puts "\t Ploting mean velocity of ions...\n"
  Gnuplot::Plot.new(gp) do |plot|
    plot.terminal "epslatex size 8,6 standalone color colortext 10"
    #plot.nokey
    plot.grid
    plot.title param_meanv_i[:title]
    plot.xlabel param_meanv_i[:xlabel]
    plot.ylabel param_meanv_i[:ylabel]
    plot.output OFNAME_MEAN_VEL_I_GRAPH 
    plot.data = [
      Gnuplot::DataSet.new( [position_bin, mean_velocity_i] ) { |ds|
        ds.with = "points ps 2"
        ds.title = "PIC simulation"
        ds.linewidth = 4
        ds.linecolor = 'rgb "blue"'
      },
      Gnuplot::DataSet.new( [position_mesh, velocity_model_i] ) { |ds|
        ds.with = "lines lt 1"
        ds.title = "Fluid model prediction"
        ds.linewidth = 4
        ds.linecolor = 'rgb "blue"'
      }
    ]
  end

  # Flux of electrons 
  puts "\t Ploting electrons's flux...\n"
  Gnuplot::Plot.new(gp) do |plot|
    plot.terminal "epslatex size 8,6 standalone color colortext 10"
    #plot.nokey
    plot.grid
    plot.title param_flux_e[:title]
    plot.xlabel param_flux_e[:xlabel]
    plot.ylabel param_flux_e[:ylabel]
    plot.output OFNAME_FLUX_E_GRAPH 
    plot.data = [
      Gnuplot::DataSet.new( [position_bin, flux_e] ) { |ds|
        ds.with = "lines lt 1"
        ds.title = "Electrons"
        ds.linewidth = 4
        ds.linecolor = 'rgb "red"'
      }
    ]
  end

  # Flux of ions
  puts "\t Ploting ions's flux...\n"
  Gnuplot::Plot.new(gp) do |plot|
    plot.terminal "epslatex size 8,6 standalone color colortext 10"
    #plot.nokey
    plot.grid
    plot.title param_flux_i[:title]
    plot.xlabel param_flux_i[:xlabel]
    plot.ylabel param_flux_i[:ylabel]
    plot.output OFNAME_FLUX_I_GRAPH 
    plot.data = [
      Gnuplot::DataSet.new( [position_bin, flux_i] ) { |ds|
        ds.with = "lines lt 1"
        ds.title = "Ions"
        ds.linewidth = 4
        ds.linecolor = 'rgb "blue"'
      }
    ]
  end
 
end

puts "Converting outputs to pdf..."
`pdflatex #{OFNAME_E2_VS_PHI}`
`pdflatex #{OFNAME_E}`
`pdflatex #{OFNAME_PHI}`
`pdflatex #{OFNAME_DDF_GRAPH}`
`pdflatex #{OFNAME_MEAN_VEL_E_GRAPH}`
`pdflatex #{OFNAME_MEAN_VEL_I_GRAPH}`
`pdflatex #{OFNAME_FLUX_E_GRAPH}`
`pdflatex #{OFNAME_FLUX_I_GRAPH}`

`rm *.aux *.log *.tex *.eps *converted-to.pdf`
