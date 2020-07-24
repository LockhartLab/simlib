# vmd_server.tcl
# Written for LLTK by C. Lockhart

set port [lindex $argv 0]

# Define print that stores value
proc print {value} {
    return [lindex [list $value] 0]
}

# Process the command
proc process_command {channel} {
    set command [gets $channel]
    set retcode [catch "uplevel #0 $command" result]
    puts $channel "$retcode$result"
}

# Establish connection with client
proc establish_connection {channel addr port} {
    fconfigure $channel -buffering line
    fileevent $channel readable [list process_command $channel]
}   

# Start server
socket -server establish_connection $port